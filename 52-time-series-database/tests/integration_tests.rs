//! Integration tests for the time-series database

use time_series_database::*;
use time_series_database::compression::*;
use time_series_database::query::*;
use time_series_database::retention::*;
use time_series_database::storage::*;
use time_series_database::wal::*;
use time_series_database::types::*;
use time_series_database::database::DatabaseConfig;
use time_series_database::storage::shard::ShardManager;

use tempfile::tempdir;
use std::collections::BTreeMap;

// ============================================================================
// Compression Tests
// ============================================================================

mod compression_tests {
    use super::*;
    use time_series_database::compression::delta::*;
    use time_series_database::compression::gorilla::*;
    use time_series_database::compression::rle::*;
    use time_series_database::compression::dictionary::*;
    use time_series_database::compression::block::*;
    use time_series_database::compression::varint::*;

    // Varint tests
    #[test]
    fn test_varint_zero() {
        let encoded = encode_varint(0);
        let (decoded, len) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, 0);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_varint_max_u64() {
        let encoded = encode_varint(u64::MAX);
        let (decoded, _) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, u64::MAX);
    }

    #[test]
    fn test_varint_powers_of_two() {
        for i in 0..64 {
            let value = 1u64 << i;
            let encoded = encode_varint(value);
            let (decoded, _) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_signed_varint_negative() {
        for value in [-1i64, -100, -1000, -10000, i64::MIN] {
            let encoded = encode_signed_varint(value);
            let (decoded, _) = decode_signed_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_signed_varint_alternating() {
        for i in 0..100 {
            let value = if i % 2 == 0 { i as i64 } else { -(i as i64) };
            let encoded = encode_signed_varint(value);
            let (decoded, _) = decode_signed_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    // Delta encoding tests
    #[test]
    fn test_delta_constant_intervals() {
        let timestamps: Vec<i64> = (0..1000).map(|i| i * 60_000_000_000).collect();
        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();
        assert_eq!(timestamps, decompressed);

        // Should be very well compressed due to constant intervals
        assert!(compressed.len() < timestamps.len() * 2);
    }

    #[test]
    fn test_delta_varying_intervals() {
        let timestamps: Vec<i64> = vec![
            1000, 1060, 1150, 1200, 1400, 1450, 1600, 1900, 2000, 2500,
        ];
        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();
        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_large_gaps() {
        let timestamps: Vec<i64> = vec![
            0,
            1_000_000_000_000,
            2_000_000_000_000,
            3_000_000_000_000,
        ];
        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();
        assert_eq!(timestamps, decompressed);
    }

    // Gorilla compression tests
    #[test]
    fn test_gorilla_constant_values() {
        let values = vec![42.5f64; 1000];
        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();
        assert_eq!(values, decompressed);

        // Should be very well compressed
        assert!(compressed.len() < 200);
    }

    #[test]
    fn test_gorilla_slowly_changing() {
        let values: Vec<f64> = (0..1000).map(|i| 50.0 + (i as f64) * 0.001).collect();
        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gorilla_sinusoidal() {
        let values: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.1).sin() * 100.0)
            .collect();
        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gorilla_special_values() {
        let values = vec![
            0.0, -0.0, 1.0, -1.0,
            f64::INFINITY, f64::NEG_INFINITY,
            f64::MIN, f64::MAX,
            f64::MIN_POSITIVE,
        ];
        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    // RLE tests
    #[test]
    fn test_rle_long_runs() {
        let values = vec![1u64; 10000];
        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(values, decompressed);
        assert!(compressed.len() < 20);
    }

    #[test]
    fn test_rle_alternating() {
        let values: Vec<u64> = (0..100).map(|i| i % 2).collect();
        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_mixed_runs() {
        let mut values = Vec::new();
        for i in 0..10 {
            for _ in 0..((i + 1) * 10) {
                values.push(i as u64);
            }
        }
        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(values, decompressed);
    }

    // Dictionary encoding tests
    #[test]
    fn test_dictionary_repeated_strings() {
        let values: Vec<&str> = vec!["status_ok"; 1000];
        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        let expected: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        assert_eq!(expected, decompressed);

        // Should be compressed
        let original_size = values.len() * values[0].len();
        assert!(compressed.len() < original_size);
    }

    #[test]
    fn test_dictionary_unique_strings() {
        let values: Vec<String> = (0..100).map(|i| format!("value_{}", i)).collect();
        let value_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let compressed = dict_compress(&value_refs);
        let decompressed = dict_decompress(&compressed).unwrap();
        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_dictionary_empty_strings() {
        let values: Vec<&str> = vec!["", "a", "", "b", ""];
        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        let expected: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        assert_eq!(expected, decompressed);
    }

    // Block compression tests
    #[test]
    fn test_block_roundtrip() {
        let points: Vec<DataPoint> = (0..500)
            .map(|i| DataPoint::new(1000 + i * 60, (i as f64).sin() * 100.0))
            .collect();

        let block = CompressedBlock::from_points(&points).unwrap();
        let decompressed = block.decompress().unwrap();

        assert_eq!(points.len(), decompressed.len());
        for (orig, dec) in points.iter().zip(decompressed.iter()) {
            assert_eq!(orig.timestamp, dec.timestamp);
            assert!((orig.value - dec.value).abs() < 1e-10);
        }
    }

    #[test]
    fn test_block_statistics() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 50.0),
            DataPoint::new(300, 30.0),
            DataPoint::new(400, 20.0),
        ];

        let block = CompressedBlock::from_points(&points).unwrap();

        assert_eq!(block.min_timestamp, 100);
        assert_eq!(block.max_timestamp, 400);
        assert_eq!(block.min_value, 10.0);
        assert_eq!(block.max_value, 50.0);
        assert_eq!(block.count, 4);
    }

    #[test]
    fn test_block_serialize_deserialize() {
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i * 60, i as f64))
            .collect();

        let block = CompressedBlock::from_points(&points).unwrap();
        let serialized = block.serialize();
        let (deserialized, _) = CompressedBlock::deserialize(&serialized).unwrap();

        assert_eq!(block.count, deserialized.count);
        assert_eq!(block.checksum, deserialized.checksum);
    }

    #[test]
    fn test_block_compressor() {
        let mut compressor = BlockCompressor::with_block_size(100);

        for i in 0..350 {
            compressor.push(DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let blocks = compressor.finish().unwrap();
        assert_eq!(blocks.len(), 4); // 100 + 100 + 100 + 50

        let mut total = 0;
        for block in &blocks {
            total += block.count as usize;
        }
        assert_eq!(total, 350);
    }

    // Combined compression tests
    #[test]
    fn test_compress_decompress_points() {
        let points: Vec<DataPoint> = (0..1000)
            .map(|i| DataPoint::new(1000000 + i * 60000, 50.0 + (i as f64 * 0.01).sin()))
            .collect();

        let compressed = compress_points(&points).unwrap();
        let decompressed = decompress_points(&compressed).unwrap();

        assert_eq!(points.len(), decompressed.len());
        for (orig, dec) in points.iter().zip(decompressed.iter()) {
            assert_eq!(orig.timestamp, dec.timestamp);
            assert!((orig.value - dec.value).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let points: Vec<DataPoint> = (0..10000)
            .map(|i| DataPoint::new(1000000 + i * 60000, 50.0))
            .collect();

        let original_size = points.len() * std::mem::size_of::<DataPoint>();
        let compressed = compress_points(&points).unwrap();
        let ratio = compression_ratio(original_size, compressed.len());

        // Constant values should compress very well
        assert!(ratio > 10.0, "Ratio was only {:.2}x", ratio);
    }
}

// ============================================================================
// Storage Tests
// ============================================================================

mod storage_tests {
    use super::*;

    #[test]
    fn test_memtable_basic_operations() {
        let memtable = MemTable::new();
        let metric = Metric::new("test").tag("host", "server1");

        for i in 0..100 {
            memtable.insert(&metric, DataPoint::new(i * 60, i as f64)).unwrap();
        }

        assert_eq!(memtable.point_count(), 100);
        assert_eq!(memtable.series_count(), 1);

        let points = memtable.query_range(metric.series_key(), 0, 3000);
        assert_eq!(points.len(), 51); // inclusive range [0, 3000]
    }

    #[test]
    fn test_memtable_multiple_series() {
        let memtable = MemTable::new();

        for i in 0..10 {
            let metric = Metric::new(format!("metric_{}", i));
            memtable.insert(&metric, DataPoint::new(100, i as f64)).unwrap();
        }

        assert_eq!(memtable.series_count(), 10);
    }

    #[test]
    fn test_memtable_out_of_order_inserts() {
        let memtable = MemTable::new();
        let metric = Metric::new("test");

        memtable.insert(&metric, DataPoint::new(300, 3.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(100, 1.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(200, 2.0)).unwrap();

        let points = memtable.query_series(metric.series_key()).unwrap();
        assert_eq!(points[0].timestamp, 100);
        assert_eq!(points[1].timestamp, 200);
        assert_eq!(points[2].timestamp, 300);
    }

    #[test]
    fn test_memtable_batch_insert() {
        let memtable = MemTable::new();
        let metric = Metric::new("test");

        let batch: Vec<_> = (0..1000)
            .map(|i| (metric.clone(), DataPoint::new(i * 60, i as f64)))
            .collect();

        memtable.insert_batch(&batch).unwrap();
        assert_eq!(memtable.point_count(), 1000);
    }

    #[test]
    fn test_sstable_create_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut series = Series::new(Metric::new("test").tag("env", "prod"));
        for i in 0..100 {
            series.push(DataPoint::new(i * 60, i as f64));
        }

        let meta = SSTable::create(&path, &[series.clone()]).unwrap();
        assert_eq!(meta.series_count, 1);
        assert_eq!(meta.point_count, 100);

        let sstable = SSTable::open(&path).unwrap();
        let read_series = sstable.read_series(series.key).unwrap().unwrap();
        assert_eq!(read_series.len(), 100);
    }

    #[test]
    fn test_sstable_query_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut series = Series::new(Metric::new("test"));
        for i in 0..1000 {
            series.push(DataPoint::new(i * 60, i as f64));
        }

        SSTable::create(&path, &[series.clone()]).unwrap();
        let sstable = SSTable::open(&path).unwrap();

        let points = sstable.query_range(series.key, 30000, 60000).unwrap();
        assert!(!points.is_empty());
        assert!(points.iter().all(|p| p.timestamp >= 30000 && p.timestamp <= 60000));
    }

    #[test]
    fn test_sstable_multiple_series() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series: Vec<_> = (0..10)
            .map(|i| {
                let mut s = Series::new(Metric::new(format!("metric_{}", i)));
                for j in 0..100 {
                    s.push(DataPoint::new(j * 60, (i * j) as f64));
                }
                s
            })
            .collect();

        let meta = SSTable::create(&path, &series).unwrap();
        assert_eq!(meta.series_count, 10);

        let sstable = SSTable::open(&path).unwrap();
        assert_eq!(sstable.series_keys().len(), 10);
    }

    #[test]
    fn test_time_shard() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            shard.insert(&metric, DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let points = shard.query_range(metric.series_key(), 0, 3600).unwrap();
        assert_eq!(points.len(), 61); // inclusive range [0, 3600]
    }

    #[test]
    fn test_time_shard_flush() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            shard.insert(&metric, DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let meta = shard.flush().unwrap();
        assert!(meta.is_some());

        // Data should still be queryable
        let points = shard.query_range(metric.series_key(), 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_shard_manager() {
        let dir = tempdir().unwrap();
        let manager = ShardManager::new(dir.path(), duration::HOUR).unwrap();

        let metric = Metric::new("test");

        // Insert data spanning multiple hours
        for hour in 0..3 {
            for minute in 0..60 {
                let ts = hour * duration::HOUR + minute * duration::MINUTE;
                manager.insert(&metric, DataPoint::new(ts, (hour * 60 + minute) as f64)).unwrap();
            }
        }

        assert_eq!(manager.shard_count(), 3);

        // Query across shards
        let points = manager.query_range(
            metric.series_key(),
            30 * duration::MINUTE,
            90 * duration::MINUTE,
        ).unwrap();

        assert!(!points.is_empty());
    }

    #[test]
    fn test_storage_engine() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        for i in 0..100 {
            engine.write("cpu.usage", &tags, i * 60, i as f64).unwrap();
        }

        let metric = Metric::with_tags("cpu.usage", tags);
        let points = engine.query_range(metric.series_key(), 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_storage_engine_batch_write() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let tags = Tags::new();
        let batch: Vec<_> = (0..1000)
            .map(|i| ("test".to_string(), tags.clone(), i * 60, i as f64))
            .collect();

        engine.write_batch(&batch).unwrap();

        let metric = Metric::with_tags("test", tags);
        let stats = engine.stats();
        assert_eq!(stats.series_count, 1);
    }
}

// ============================================================================
// Query Engine Tests
// ============================================================================

mod query_tests {
    use super::*;

    fn create_test_points(count: usize) -> Vec<DataPoint> {
        (0..count)
            .map(|i| DataPoint::new(i as i64 * 60, i as f64))
            .collect()
    }

    #[test]
    fn test_aggregation_sum() {
        let points = create_test_points(100);
        let result = aggregation::aggregate(&points, Aggregation::Sum);
        assert_eq!(result, (0..100).sum::<i32>() as f64);
    }

    #[test]
    fn test_aggregation_avg() {
        let points = create_test_points(100);
        let result = aggregation::aggregate(&points, Aggregation::Avg);
        assert_eq!(result, 49.5);
    }

    #[test]
    fn test_aggregation_min_max() {
        let points = create_test_points(100);
        assert_eq!(aggregation::aggregate(&points, Aggregation::Min), 0.0);
        assert_eq!(aggregation::aggregate(&points, Aggregation::Max), 99.0);
    }

    #[test]
    fn test_aggregation_first_last() {
        let points = create_test_points(100);
        assert_eq!(aggregation::aggregate(&points, Aggregation::First), 0.0);
        assert_eq!(aggregation::aggregate(&points, Aggregation::Last), 99.0);
    }

    #[test]
    fn test_aggregation_count() {
        let points = create_test_points(100);
        assert_eq!(aggregation::aggregate(&points, Aggregation::Count), 100.0);
    }

    #[test]
    fn test_aggregation_percentile() {
        let points: Vec<DataPoint> = (1..=100)
            .map(|i| DataPoint::new(i, i as f64))
            .collect();

        let p50 = aggregation::aggregate(&points, Aggregation::Percentile(50));
        let p90 = aggregation::aggregate(&points, Aggregation::Percentile(90));

        assert!((p50 - 50.0).abs() <= 2.0, "p50 was {}", p50);
        assert!((p90 - 90.0).abs() <= 2.0, "p90 was {}", p90);
    }

    #[test]
    fn test_aggregation_with_interval() {
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i * 10, i as f64))
            .collect();

        let result = aggregation::aggregate_with_interval(
            &points,
            Aggregation::Sum,
            100,
            0,
            1000,
        );

        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_predicate_value() {
        let point = DataPoint::new(100, 50.0);

        assert!(Predicate::value_gt(40.0).evaluate(&point));
        assert!(!Predicate::value_gt(60.0).evaluate(&point));
        assert!(Predicate::value_lt(60.0).evaluate(&point));
        assert!(Predicate::value_between(40.0, 60.0).evaluate(&point));
    }

    #[test]
    fn test_predicate_timestamp() {
        let point = DataPoint::new(100, 50.0);

        assert!(Predicate::timestamp_between(50, 150).evaluate(&point));
        assert!(!Predicate::timestamp_between(200, 300).evaluate(&point));
    }

    #[test]
    fn test_predicate_logical_ops() {
        let point = DataPoint::new(100, 50.0);

        let and_pred = Predicate::value_gt(40.0).and(Predicate::value_lt(60.0));
        assert!(and_pred.evaluate(&point));

        let or_pred = Predicate::value_lt(40.0).or(Predicate::value_gt(40.0));
        assert!(or_pred.evaluate(&point));

        let not_pred = Predicate::value_lt(40.0).not();
        assert!(not_pred.evaluate(&point));
    }

    #[test]
    fn test_predicate_filter() {
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i, i as f64))
            .collect();

        let pred = Predicate::value_gt(50.0);
        let filtered = pred.filter_owned(&points);

        assert_eq!(filtered.len(), 49);
        assert!(filtered.iter().all(|p| p.value > 50.0));
    }

    #[test]
    fn test_query_executor() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            engine.write_points(&[(metric.clone(), DataPoint::new(i * 60, i as f64))]).unwrap();
        }

        let executor = QueryExecutor::new(&engine);

        // Range query (inclusive range)
        let points = executor.range(metric.series_key(), 0, 3000).unwrap();
        assert_eq!(points.len(), 51);

        // Aggregate query
        let sum = executor.aggregate(metric.series_key(), 0, 6000, Aggregation::Sum).unwrap();
        assert_eq!(sum, (0..100).sum::<i32>() as f64);

        // Downsample query
        let downsampled = executor.downsample(
            metric.series_key(),
            0,
            6000,
            600,
            Aggregation::Avg,
        ).unwrap();
        assert_eq!(downsampled.len(), 10);
    }

    #[test]
    fn test_query_with_predicate() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            engine.write_points(&[(metric.clone(), DataPoint::new(i * 60, i as f64))]).unwrap();
        }

        let query = Query::new(metric.series_key(), 0, 6000)
            .with_predicate(Predicate::value_gt(50.0));

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        let series_result = result.get(metric.series_key()).unwrap();
        assert!(series_result.points.iter().all(|p| p.value > 50.0));
    }

    #[test]
    fn test_query_builder() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            engine.write_points(&[(metric.clone(), DataPoint::new(i * 60, i as f64))]).unwrap();
        }

        let query = executor::QueryBuilder::new()
            .series(metric.series_key())
            .time_range(0, 3000)
            .aggregate(Aggregation::Avg)
            .build()
            .unwrap();

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        let series_result = result.get(metric.series_key()).unwrap();
        assert!(series_result.aggregate_value.is_some());
    }
}

// ============================================================================
// Retention Tests
// ============================================================================

mod retention_tests {
    use super::*;
    use time_series_database::retention::templates;

    #[test]
    fn test_retention_policy_basic() {
        let policy = RetentionPolicy::new("test", 7 * duration::DAY);

        assert_eq!(policy.max_retention(), 7 * duration::DAY);
        assert!(policy.should_drop(8 * duration::DAY));
        assert!(!policy.should_drop(6 * duration::DAY));
    }

    #[test]
    fn test_retention_policy_with_downsample() {
        let policy = templates::standard();

        // Raw data for first 7 days
        assert!(policy.resolution_for_age(duration::DAY).is_none());

        // 1-minute data from 7-30 days
        assert_eq!(policy.resolution_for_age(10 * duration::DAY), Some(duration::MINUTE));

        // 1-hour data after 30 days
        assert_eq!(policy.resolution_for_age(60 * duration::DAY), Some(duration::HOUR));
    }

    #[test]
    fn test_downsample_rule() {
        let rule = DownsampleRule::new(
            duration::DAY,
            duration::MINUTE,
            30 * duration::DAY,
            Aggregation::Avg,
        );

        let points: Vec<DataPoint> = (0..120)
            .map(|i| DataPoint::new(i * duration::SECOND, i as f64))
            .collect();

        let downsampled = rule.apply(&points, 0, 2 * duration::MINUTE);
        assert_eq!(downsampled.len(), 2);
    }

    #[test]
    fn test_retention_manager() {
        let manager = RetentionManager::new();

        manager.add_policy(templates::standard());
        manager.add_policy(templates::long_term());

        let names = manager.policy_names();
        assert_eq!(names.len(), 2);

        let standard = manager.get_policy("standard").unwrap();
        assert_eq!(standard.name, "standard");
    }

    #[test]
    fn test_retention_templates() {
        // Short term
        let short = templates::short_term();
        assert_eq!(short.max_retention(), 7 * duration::DAY);

        // Standard
        let standard = templates::standard();
        assert!(!standard.downsample_rules.is_empty());

        // Long term
        let long = templates::long_term();
        assert!(long.max_retention() > 365 * duration::DAY);

        // High resolution
        let high = templates::high_resolution();
        assert!(high.downsample_rules.is_empty());
    }
}

// ============================================================================
// WAL Tests
// ============================================================================

mod wal_tests {
    use super::*;
    use time_series_database::wal::*;

    #[test]
    fn test_wal_entry_serialize_write() {
        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        let entry = WalEntry::Write {
            metric_name: "cpu.usage".into(),
            tags,
            timestamp: 1000,
            value: 42.5,
        };

        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        if let WalEntry::Write { metric_name, timestamp, value, .. } = deserialized {
            assert_eq!(metric_name, "cpu.usage");
            assert_eq!(timestamp, 1000);
            assert_eq!(value, 42.5);
        } else {
            panic!("Wrong entry type");
        }
    }

    #[test]
    fn test_wal_entry_serialize_batch() {
        let tags = Tags::new();
        let entry = WalEntry::WriteBatch {
            points: vec![
                ("metric1".into(), tags.clone(), 100, 1.0),
                ("metric2".into(), tags.clone(), 200, 2.0),
            ],
        };

        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        if let WalEntry::WriteBatch { points } = deserialized {
            assert_eq!(points.len(), 2);
        } else {
            panic!("Wrong entry type");
        }
    }

    #[test]
    fn test_wal_append_replay() {
        let dir = tempdir().unwrap();
        let wal = WriteAheadLog::new(dir.path()).unwrap();

        let tags = Tags::new();
        for i in 0..10 {
            wal.append(&WalEntry::Write {
                metric_name: "test".into(),
                tags: tags.clone(),
                timestamp: i * 100,
                value: i as f64,
            }).unwrap();
        }

        wal.sync().unwrap();

        let mut count = 0;
        wal.replay(|_| {
            count += 1;
            Ok(())
        }).unwrap();

        assert_eq!(count, 10);
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();

        // Write entries
        {
            let wal = WriteAheadLog::new(dir.path()).unwrap();
            let tags = Tags::new();

            for i in 0..10 {
                wal.append(&WalEntry::Write {
                    metric_name: "test".into(),
                    tags: tags.clone(),
                    timestamp: i * 100,
                    value: i as f64,
                }).unwrap();
            }

            wal.sync().unwrap();
        }

        // Recover
        {
            let wal = WriteAheadLog::new(dir.path()).unwrap();

            let mut count = 0;
            wal.replay(|_| {
                count += 1;
                Ok(())
            }).unwrap();

            assert_eq!(count, 10);
        }
    }
}

// ============================================================================
// Database Integration Tests
// ============================================================================

mod database_tests {
    use super::*;
    use time_series_database::database::*;

    fn test_config(dir: &std::path::Path) -> DatabaseConfig {
        DatabaseConfig::new(dir)
            .without_wal()
            .without_background_tasks()
    }

    #[test]
    fn test_database_write_query() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("cpu.usage", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("cpu.usage", &tags);
        let points = db.query_range(series_key, 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_database_batch_write() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        let batch: Vec<_> = (0..1000)
            .map(|i| ("test".to_string(), tags.clone(), i * 60, i as f64))
            .collect();

        db.write_batch(&batch).unwrap();

        let series_key = db.series_key("test", &tags);
        let points = db.query_range(series_key, 0, 60000).unwrap();
        assert_eq!(points.len(), 1000);
    }

    #[test]
    fn test_database_aggregation() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("test", &tags);
        let sum = db.aggregate(series_key, 0, 6000, Aggregation::Sum).unwrap();
        assert_eq!(sum, (0..100).sum::<i32>() as f64);
    }

    #[test]
    fn test_database_downsample() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("test", &tags);
        let result = db.downsample(series_key, 0, 6000, 600, Aggregation::Avg).unwrap();
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_database_find_series() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        db.write("cpu.usage", &tags, 100, 1.0).unwrap();
        db.write("cpu.system", &tags, 100, 2.0).unwrap();
        db.write("memory.used", &tags, 100, 3.0).unwrap();

        let cpu_series = db.find_series("cpu.");
        assert_eq!(cpu_series.len(), 2);
    }

    #[test]
    fn test_database_flush_compact() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        db.flush().unwrap();
        db.compact().unwrap();

        let series_key = db.series_key("test", &tags);
        let points = db.query_range(series_key, 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_database_stats() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let stats = db.stats();
        assert_eq!(stats.series_count, 0);

        let tags = Tags::new();
        db.write("test", &tags, 100, 1.0).unwrap();

        let stats = db.stats();
        assert_eq!(stats.series_count, 1);
    }

    #[test]
    fn test_database_retention() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        db.set_default_retention(RetentionPolicy::new("test", duration::DAY));

        let tags = Tags::new();
        db.write("test", &tags, 100, 1.0).unwrap();

        db.apply_retention().unwrap();
    }

    #[test]
    fn test_database_with_wal() {
        let dir = tempdir().unwrap();

        // Write data
        {
            let config = DatabaseConfig::new(dir.path()).without_background_tasks();
            let db = TimeSeriesDB::open(config).unwrap();

            let tags = Tags::new();
            for i in 0..10 {
                db.write("test", &tags, i * 60, i as f64).unwrap();
            }

            db.close().unwrap();
        }

        // Recover and verify
        {
            let config = DatabaseConfig::new(dir.path()).without_background_tasks();
            let db = TimeSeriesDB::open(config).unwrap();

            let tags = Tags::new();
            let series_key = db.series_key("test", &tags);
            let points = db.query_range(series_key, 0, 1000).unwrap();

            assert_eq!(points.len(), 10);
        }
    }

    #[test]
    fn test_database_multiple_metrics() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();

        // Write multiple metrics
        for metric in ["cpu", "memory", "disk", "network"] {
            for i in 0..100 {
                db.write(metric, &tags, i * 60, i as f64).unwrap();
            }
        }

        let stats = db.stats();
        assert_eq!(stats.series_count, 4);

        // Query each metric
        for metric in ["cpu", "memory", "disk", "network"] {
            let series_key = db.series_key(metric, &tags);
            let points = db.query_range(series_key, 0, 6000).unwrap();
            assert_eq!(points.len(), 100);
        }
    }

    #[test]
    fn test_database_with_tags() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        // Write data with different tags
        for host in ["server1", "server2", "server3"] {
            let mut tags = Tags::new();
            tags.insert("host".into(), host.into());

            for i in 0..50 {
                db.write("cpu.usage", &tags, i * 60, i as f64).unwrap();
            }
        }

        let stats = db.stats();
        assert_eq!(stats.series_count, 3);
    }

    #[test]
    fn test_database_query_by_metric() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        for i in 0..100 {
            db.write("cpu.usage", &tags, i * 60, i as f64).unwrap();
        }

        let points = db.query_metric("cpu.usage", &tags, 0, 3000).unwrap();
        assert_eq!(points.len(), 51); // inclusive range
    }
}

// ============================================================================
// Types Tests
// ============================================================================

mod types_tests {
    use super::*;

    #[test]
    fn test_data_point() {
        let point = DataPoint::new(1000, 42.5);
        assert_eq!(point.timestamp, 1000);
        assert_eq!(point.value, 42.5);
    }

    #[test]
    fn test_metric_creation() {
        let metric = Metric::new("cpu.usage")
            .tag("host", "server1")
            .tag("region", "us-east");

        assert_eq!(metric.name, "cpu.usage");
        assert_eq!(metric.tags.get("host"), Some(&"server1".to_string()));
    }

    #[test]
    fn test_series_key_consistency() {
        let metric1 = Metric::new("test").tag("a", "1").tag("b", "2");
        let metric2 = Metric::new("test").tag("a", "1").tag("b", "2");
        let metric3 = Metric::new("test").tag("b", "2").tag("a", "1");

        assert_eq!(metric1.series_key(), metric2.series_key());
        assert_eq!(metric1.series_key(), metric3.series_key());
    }

    #[test]
    fn test_series_operations() {
        let mut series = Series::new(Metric::new("test"));

        series.push(DataPoint::new(300, 3.0));
        series.push(DataPoint::new(100, 1.0));
        series.push(DataPoint::new(200, 2.0));

        series.sort();

        assert_eq!(series.first_timestamp(), Some(100));
        assert_eq!(series.last_timestamp(), Some(300));

        let range = series.range(100, 200);
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_write_batch() {
        let mut batch = WriteBatch::new();

        batch.push(Metric::new("test1"), DataPoint::new(100, 1.0));
        batch.push(Metric::new("test2"), DataPoint::new(200, 2.0));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_duration_constants() {
        assert_eq!(duration::SECOND, 1_000_000_000);
        assert_eq!(duration::MINUTE, 60 * duration::SECOND);
        assert_eq!(duration::HOUR, 60 * duration::MINUTE);
        assert_eq!(duration::DAY, 24 * duration::HOUR);
        assert_eq!(duration::WEEK, 7 * duration::DAY);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_high_cardinality_series() {
        let dir = tempdir().unwrap();
        let config = DatabaseConfig::new(dir.path())
            .without_wal()
            .without_background_tasks();
        let db = TimeSeriesDB::open(config).unwrap();

        // Create 100 different series
        for i in 0..100 {
            let mut tags = Tags::new();
            tags.insert("host".into(), format!("server{}", i));

            for j in 0..10 {
                db.write("cpu.usage", &tags, j * 60, j as f64).unwrap();
            }
        }

        let stats = db.stats();
        assert_eq!(stats.series_count, 100);
    }

    #[test]
    fn test_large_batch_write() {
        let dir = tempdir().unwrap();
        let config = DatabaseConfig::new(dir.path())
            .without_wal()
            .without_background_tasks();
        let db = TimeSeriesDB::open(config).unwrap();

        let tags = Tags::new();
        let batch: Vec<_> = (0..10000)
            .map(|i| ("test".to_string(), tags.clone(), i * 60, i as f64))
            .collect();

        db.write_batch(&batch).unwrap();

        let series_key = db.series_key("test", &tags);
        let points = db.query_range(series_key, 0, 600000).unwrap();
        assert_eq!(points.len(), 10000);
    }

    #[test]
    fn test_compression_efficiency() {
        // Monotonically increasing timestamps with constant values
        let points: Vec<DataPoint> = (0..10000)
            .map(|i| DataPoint::new(i * 1000, 50.0))
            .collect();

        let original_size = points.len() * std::mem::size_of::<DataPoint>();
        let compressed = compress_points(&points).unwrap();
        let ratio = original_size as f64 / compressed.len() as f64;

        // Should achieve good compression with constant values
        assert!(ratio > 10.0, "Compression ratio was only {:.2}x", ratio);
    }
}
