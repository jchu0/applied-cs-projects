//! Comprehensive tests for metrics module.

use simd_analytics_engine::metrics::{
    PerfMetrics, QueryMetrics, Timer, Benchmark, BenchmarkResult, PerformanceModel,
};
use std::time::Duration;

// ============================================================================
// PerfMetrics Tests
// ============================================================================

#[test]
fn test_perf_metrics_calculate() {
    let metrics = PerfMetrics::calculate(
        Duration::from_millis(100),
        1_000_000,
        8_000_000,
    );

    assert_eq!(metrics.rows_processed, 1_000_000);
    assert_eq!(metrics.bytes_processed, 8_000_000);
    assert_eq!(metrics.rows_per_second, 10_000_000.0);
    assert!((metrics.gb_per_second - 0.08).abs() < 0.001);
}

#[test]
fn test_perf_metrics_calculate_zero_time() {
    let metrics = PerfMetrics::calculate(
        Duration::ZERO,
        1000,
        8000,
    );

    assert_eq!(metrics.rows_per_second, 0.0);
    assert_eq!(metrics.gb_per_second, 0.0);
}

#[test]
fn test_perf_metrics_format() {
    let metrics = PerfMetrics::calculate(
        Duration::from_millis(100),
        1_000_000,
        8_000_000,
    );

    let formatted = metrics.format();
    assert!(formatted.contains("Time:"));
    assert!(formatted.contains("Rows:"));
}

// ============================================================================
// QueryMetrics Tests
// ============================================================================

#[test]
fn test_query_metrics_new() {
    let metrics = QueryMetrics::new();

    assert_eq!(metrics.total_time, Duration::ZERO);
    assert_eq!(metrics.rows_scanned, 0);
}

#[test]
fn test_query_metrics_throughput() {
    let mut metrics = QueryMetrics::new();
    metrics.total_time = Duration::from_secs(1);
    metrics.rows_scanned = 1_000_000;

    assert_eq!(metrics.throughput(), 1_000_000.0);
}

#[test]
fn test_query_metrics_throughput_zero_time() {
    let mut metrics = QueryMetrics::new();
    metrics.total_time = Duration::ZERO;
    metrics.rows_scanned = 1_000_000;

    assert_eq!(metrics.throughput(), 0.0);
}

#[test]
fn test_query_metrics_selectivity() {
    let mut metrics = QueryMetrics::new();
    metrics.rows_scanned = 1000;
    metrics.rows_filtered = 100;

    assert!((metrics.selectivity() - 0.1).abs() < 0.001);
}

#[test]
fn test_query_metrics_selectivity_zero_scanned() {
    let mut metrics = QueryMetrics::new();
    metrics.rows_scanned = 0;
    metrics.rows_filtered = 100;

    assert_eq!(metrics.selectivity(), 0.0);
}

#[test]
fn test_query_metrics_format_dashboard() {
    let mut metrics = QueryMetrics::new();
    metrics.total_time = Duration::from_millis(100);
    metrics.rows_scanned = 10000;
    metrics.rows_filtered = 1000;
    metrics.bytes_scanned = 80000;

    let dashboard = metrics.format_dashboard();

    assert!(dashboard.contains("Query Performance Dashboard"));
    assert!(dashboard.contains("Total Time:"));
}

// ============================================================================
// Timer Tests
// ============================================================================

#[test]
fn test_timer_start() {
    let timer = Timer::start();
    std::thread::sleep(Duration::from_millis(10));

    let elapsed = timer.elapsed();
    assert!(elapsed >= Duration::from_millis(10));
}

#[test]
fn test_timer_split() {
    let mut timer = Timer::start();

    std::thread::sleep(Duration::from_millis(10));
    timer.split("phase1");

    std::thread::sleep(Duration::from_millis(10));
    timer.split("phase2");

    assert_eq!(timer.splits().len(), 2);
}

#[test]
fn test_timer_splits() {
    let mut timer = Timer::start();

    std::thread::sleep(Duration::from_millis(5));
    timer.split("first");

    std::thread::sleep(Duration::from_millis(5));
    timer.split("second");

    let splits = timer.splits();
    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].0, "first");
    assert_eq!(splits[1].0, "second");
}

#[test]
fn test_timer_format_splits() {
    let mut timer = Timer::start();

    std::thread::sleep(Duration::from_millis(5));
    timer.split("phase1");

    let formatted = timer.format_splits();
    assert!(formatted.contains("phase1"));
}

// ============================================================================
// Benchmark Tests
// ============================================================================

#[test]
fn test_benchmark_default() {
    let benchmark = Benchmark::default();
    assert!(benchmark.warmup_iters > 0);
    assert!(benchmark.measure_iters > 0);
}

#[test]
fn test_benchmark_new() {
    let benchmark = Benchmark::new(2, 5);
    assert_eq!(benchmark.warmup_iters, 2);
    assert_eq!(benchmark.measure_iters, 5);
}

#[test]
fn test_benchmark_run() {
    let benchmark = Benchmark::new(1, 3);

    let mut counter = 0;
    let result = benchmark.run(|| {
        counter += 1;
        std::thread::sleep(Duration::from_millis(1));
    });

    // 1 warmup + 3 measure = 4 total calls
    assert_eq!(counter, 4);
    assert_eq!(result.times.len(), 3);
}

#[test]
fn test_benchmark_run_with_setup() {
    let benchmark = Benchmark::new(1, 3);

    let result = benchmark.run_with_setup(
        || vec![1, 2, 3, 4, 5],
        |input| {
            let _: i32 = input.iter().sum();
        },
    );

    assert_eq!(result.times.len(), 3);
}

// ============================================================================
// BenchmarkResult Tests
// ============================================================================

#[test]
fn test_benchmark_result_from_times() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(11),
        Duration::from_millis(12),
        Duration::from_millis(9),
        Duration::from_millis(13),
    ];

    let result = BenchmarkResult::from_times(times);

    assert_eq!(result.min_time, Duration::from_millis(9));
    assert_eq!(result.max_time, Duration::from_millis(13));
    assert_eq!(result.median_time, Duration::from_millis(11));
}

#[test]
fn test_benchmark_result_format() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(11),
        Duration::from_millis(12),
    ];

    let result = BenchmarkResult::from_times(times);
    let formatted = result.format();

    assert!(formatted.contains("min:"));
    assert!(formatted.contains("median:"));
    assert!(formatted.contains("mean:"));
    assert!(formatted.contains("max:"));
}

#[test]
fn test_benchmark_result_mean() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(30),
    ];

    let result = BenchmarkResult::from_times(times);

    // Mean should be 20ms
    assert_eq!(result.mean_time, Duration::from_millis(20));
}

// ============================================================================
// PerformanceModel Tests
// ============================================================================

#[test]
fn test_performance_model_default() {
    let model = PerformanceModel::default();

    assert!(model.memory_bandwidth > 0.0);
    assert!(model.l1_cache_size > 0);
    assert!(model.cpu_freq > 0.0);
}

#[test]
fn test_performance_model_estimate_memory_bound() {
    let model = PerformanceModel::default();

    let bytes = 50_000_000_000; // 50GB
    let time = model.estimate_memory_bound(bytes);

    // With 50 GB/s bandwidth, 50GB should take ~1 second
    assert!((time.as_secs_f64() - 1.0).abs() < 0.1);
}

#[test]
fn test_performance_model_estimate_compute_bound() {
    let model = PerformanceModel::default();

    let ops = 3_000_000_000; // 3 billion ops
    let time = model.estimate_compute_bound(ops);

    // Should estimate some non-zero time
    assert!(time.as_secs_f64() > 0.0);
}

#[test]
fn test_performance_model_estimate_scan() {
    let model = PerformanceModel::default();

    let time = model.estimate_scan(1_000_000, 8);

    // Should estimate some non-zero time
    assert!(time.as_secs_f64() > 0.0);
}

#[test]
fn test_performance_model_estimate_aggregate() {
    let model = PerformanceModel::default();

    let time = model.estimate_aggregate(1_000_000);

    // Should estimate some non-zero time
    assert!(time.as_secs_f64() > 0.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_benchmarking() {
    let benchmark = Benchmark::new(2, 5);

    let result = benchmark.run(|| {
        let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let _sum: f64 = data.iter().sum();
    });

    assert!(result.min_time < result.max_time || result.min_time == result.max_time);
    assert!(result.mean_time >= result.min_time);
    assert!(result.mean_time <= result.max_time);
}

#[test]
fn test_timer_with_multiple_phases() {
    let mut timer = Timer::start();

    // Phase 1: Setup
    let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    timer.split("setup");

    // Phase 2: Processing
    let _sum: f64 = data.iter().sum();
    timer.split("processing");

    // Phase 3: Cleanup
    drop(data);
    timer.split("cleanup");

    assert_eq!(timer.splits().len(), 3);
    assert!(timer.elapsed() >= timer.splits().iter().map(|(_, d)| *d).sum());
}

#[test]
fn test_performance_model_comparisons() {
    let model = PerformanceModel::default();

    let small_scan = model.estimate_scan(1000, 8);
    let large_scan = model.estimate_scan(1_000_000, 8);

    // Larger scan should take more time
    assert!(large_scan > small_scan);
}
