use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use time_series_database::*;
use time_series_database::compression::*;
use time_series_database::types::*;
use tempfile::tempdir;

fn generate_points(count: usize) -> Vec<DataPoint> {
    (0..count)
        .map(|i| DataPoint::new(1000000000 + i as i64 * 60000000000, (i as f64 * 0.1).sin() * 100.0))
        .collect()
}

fn compression_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    for size in [100, 1000, 10000].iter() {
        let points = generate_points(*size);

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, points| {
            b.iter(|| compress_points(black_box(points)))
        });

        let compressed = compress_points(&points).unwrap();
        group.bench_with_input(BenchmarkId::new("decompress", size), &compressed, |b, data| {
            b.iter(|| decompress_points(black_box(data)))
        });
    }

    group.finish();
}

fn database_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("database");

    // Single write benchmark
    group.bench_function("single_write", |b| {
        let dir = tempdir().unwrap();
        let config = database::DatabaseConfig::new(dir.path())
            .without_wal()
            .without_background_tasks();
        let db = TimeSeriesDB::open(config).unwrap();
        let tags = Tags::new();

        let mut i = 0i64;
        b.iter(|| {
            db.write(black_box("test"), black_box(&tags), i * 60, i as f64).unwrap();
            i += 1;
        });
    });

    // Batch write benchmark
    for batch_size in [100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_write", batch_size),
            batch_size,
            |b, &size| {
                let dir = tempdir().unwrap();
                let config = database::DatabaseConfig::new(dir.path())
                    .without_wal()
                    .without_background_tasks();
                let db = TimeSeriesDB::open(config).unwrap();
                let tags = Tags::new();

                let batch: Vec<_> = (0..size)
                    .map(|i| ("test".to_string(), tags.clone(), i as i64 * 60, i as f64))
                    .collect();

                b.iter(|| {
                    db.write_batch(black_box(&batch)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn query_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    // Setup database with data
    let dir = tempdir().unwrap();
    let config = database::DatabaseConfig::new(dir.path())
        .without_wal()
        .without_background_tasks();
    let db = TimeSeriesDB::open(config).unwrap();
    let tags = Tags::new();

    // Insert test data
    let batch: Vec<_> = (0..10000)
        .map(|i| ("test".to_string(), tags.clone(), i as i64 * 60, i as f64))
        .collect();
    db.write_batch(&batch).unwrap();

    let series_key = db.series_key("test", &tags);

    // Range query benchmark
    group.bench_function("range_query", |b| {
        b.iter(|| {
            db.query_range(black_box(series_key), black_box(0), black_box(60000)).unwrap()
        });
    });

    // Aggregation benchmark
    group.bench_function("aggregation_sum", |b| {
        b.iter(|| {
            db.aggregate(black_box(series_key), black_box(0), black_box(600000), black_box(query::Aggregation::Sum)).unwrap()
        });
    });

    // Downsampling benchmark
    group.bench_function("downsample", |b| {
        b.iter(|| {
            db.downsample(
                black_box(series_key),
                black_box(0),
                black_box(600000),
                black_box(60000),
                black_box(query::Aggregation::Avg),
            ).unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, compression_benchmarks, database_benchmarks, query_benchmarks);
criterion_main!(benches);
