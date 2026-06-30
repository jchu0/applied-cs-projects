//! Tests for the metrics module.

use gpu_gemm_optimization::{Matrix, naive_gemm, GemmMetrics};
use gpu_gemm_optimization::metrics::{
    Benchmark, BenchmarkResults, PerformanceModel, KernelComparison, MemoryProfile,
};
use std::time::Duration;

#[test]
fn test_gemm_metrics_calculate() {
    let metrics = GemmMetrics::calculate(
        1024, 1024, 1024,
        Duration::from_millis(100),
        100.0,
    );

    // 2 * 1024^3 = 2^31 FLOP
    // In 100ms = 0.1s
    // GFLOPS = 2^31 / (0.1 * 10^9) ≈ 21.5
    assert!(metrics.gflops > 20.0 && metrics.gflops < 25.0);
    assert!(metrics.bandwidth_gbs > 0.0);
    assert!(metrics.arithmetic_intensity > 0.0);
}

#[test]
fn test_gemm_metrics_format() {
    let metrics = GemmMetrics::calculate(
        256, 256, 256,
        Duration::from_millis(10),
        100.0,
    );

    let formatted = metrics.format();
    assert!(formatted.contains("256x256x256"));
    assert!(formatted.contains("GFLOPS"));
    assert!(formatted.contains("GB/s"));
}

#[test]
fn test_benchmark_default() {
    let bench = Benchmark::default();

    assert_eq!(bench.warmup_iters, 3);
    assert_eq!(bench.measure_iters, 10);
    assert_eq!(bench.peak_gflops, 100.0);
}

#[test]
fn test_benchmark_new() {
    let bench = Benchmark::new(5, 20, 200.0);

    assert_eq!(bench.warmup_iters, 5);
    assert_eq!(bench.measure_iters, 20);
    assert_eq!(bench.peak_gflops, 200.0);
}

#[test]
fn test_benchmark_run() {
    let bench = Benchmark::new(2, 5, 100.0);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = bench.run(&a, &b, |a, b, c| naive_gemm(a, b, c));

    assert!(result.is_ok());
    let metrics = result.unwrap();

    assert!(metrics.gflops > 0.0);
    assert!(metrics.execution_time.as_nanos() > 0);
    assert_eq!(metrics.dimensions, (64, 64, 64));
}

#[test]
fn test_benchmark_run_detailed() {
    let bench = Benchmark::new(2, 5, 100.0);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = bench.run_detailed(&a, &b, |a, b, c| naive_gemm(a, b, c));

    assert!(result.is_ok());
    let results = result.unwrap();

    assert_eq!(results.times.len(), 5);
    assert!(results.min_time <= results.median_time);
    assert!(results.median_time <= results.max_time);
}

#[test]
fn test_benchmark_results_statistics() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(11),
        Duration::from_millis(12),
        Duration::from_millis(9),
        Duration::from_millis(13),
    ];

    let results = BenchmarkResults::new(128, 128, 128, times, 100.0);

    assert_eq!(results.min_time, Duration::from_millis(9));
    assert_eq!(results.max_time, Duration::from_millis(13));
    assert_eq!(results.median_time, Duration::from_millis(11));
    assert!(results.best_gflops > results.median_gflops);
}

#[test]
fn test_benchmark_results_cv() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(10),
        Duration::from_millis(10),
    ];

    let results = BenchmarkResults::new(64, 64, 64, times, 100.0);

    // With identical times, CV should be close to 0
    assert!(results.cv() < 0.01);
}

#[test]
fn test_benchmark_results_format() {
    let times = vec![
        Duration::from_millis(10),
        Duration::from_millis(11),
        Duration::from_millis(12),
    ];

    let results = BenchmarkResults::new(128, 128, 128, times, 100.0);
    let formatted = results.format();

    assert!(formatted.contains("128x128x128"));
    assert!(formatted.contains("Best"));
    assert!(formatted.contains("Median"));
    assert!(formatted.contains("Mean"));
}

#[test]
fn test_performance_model_default() {
    let model = PerformanceModel::default();

    assert_eq!(model.memory_bandwidth, 50.0);
    assert_eq!(model.peak_compute, 100.0);
}

#[test]
fn test_performance_model_new() {
    let model = PerformanceModel::new(100.0, 500.0);

    assert_eq!(model.memory_bandwidth, 100.0);
    assert_eq!(model.peak_compute, 500.0);
}

#[test]
fn test_performance_model_ridge_point() {
    let model = PerformanceModel::new(50.0, 100.0);

    // Ridge point = 100 / 50 = 2 FLOP/byte
    assert!((model.ridge_point() - 2.0).abs() < 0.01);
}

#[test]
fn test_performance_model_memory_bound() {
    let model = PerformanceModel::new(50.0, 100.0);

    // Below ridge point (2.0) = memory bound
    assert!(model.is_memory_bound(1.0));
    assert!(model.is_memory_bound(1.5));

    // Above ridge point = compute bound
    assert!(!model.is_memory_bound(3.0));
    assert!(!model.is_memory_bound(10.0));
}

#[test]
fn test_performance_model_predict_gflops() {
    let model = PerformanceModel::new(50.0, 100.0);

    // Memory bound case: prediction = bandwidth * intensity
    assert_eq!(model.predict_gflops(1.0), 50.0);

    // Compute bound case: prediction = peak
    assert_eq!(model.predict_gflops(3.0), 100.0);
}

#[test]
fn test_performance_model_arithmetic_intensity() {
    // For 1024x1024x1024 GEMM:
    // FLOPS = 2 * 1024^3
    // Bytes = (1024^2 + 1024^2 + 1024^2) * 4 = 3 * 1024^2 * 4
    // AI = 2 * 1024^3 / (3 * 1024^2 * 4) = 2 * 1024 / 12 ≈ 170
    let intensity = PerformanceModel::gemm_arithmetic_intensity(1024, 1024, 1024);
    assert!(intensity > 100.0);
}

#[test]
fn test_performance_model_predict_gemm() {
    let model = PerformanceModel::new(50.0, 100.0);

    // Large GEMM should be compute-bound
    let prediction = model.predict_gemm(1024, 1024, 1024);
    assert_eq!(prediction, 100.0); // Should hit peak

    // Small GEMM might be memory-bound
    let prediction_small = model.predict_gemm(16, 16, 16);
    assert!(prediction_small <= 100.0);
}

#[test]
fn test_performance_model_optimal_tile_size() {
    let model = PerformanceModel::default();

    let l1_tile = model.optimal_tile_size(1);
    let l2_tile = model.optimal_tile_size(2);
    let l3_tile = model.optimal_tile_size(3);

    // Larger caches should support larger tiles
    assert!(l1_tile < l2_tile);
    assert!(l2_tile < l3_tile);
}

#[test]
fn test_kernel_comparison() {
    let mut comparison = KernelComparison::new("baseline".to_string());

    let times1 = vec![Duration::from_millis(10); 5];
    let times2 = vec![Duration::from_millis(8); 5];

    let results1 = BenchmarkResults::new(128, 128, 128, times1, 100.0);
    let results2 = BenchmarkResults::new(128, 128, 128, times2, 100.0);

    comparison.add("baseline".to_string(), results1);
    comparison.add("optimized".to_string(), results2);

    // Optimized should be faster
    let speedup = comparison.speedup("optimized");
    assert!(speedup.is_some());
    assert!(speedup.unwrap() > 1.0);
}

#[test]
fn test_kernel_comparison_format() {
    let mut comparison = KernelComparison::new("naive".to_string());

    let times = vec![Duration::from_millis(10); 5];
    let results = BenchmarkResults::new(128, 128, 128, times, 100.0);

    comparison.add("naive".to_string(), results);

    let formatted = comparison.format();
    assert!(formatted.contains("Kernel Comparison"));
    assert!(formatted.contains("naive"));
    assert!(formatted.contains("GFLOPS"));
}

#[test]
fn test_memory_profile_hit_rate() {
    let mut profile = MemoryProfile::default();
    profile.total_accesses = 1000;
    profile.cache_hits = 900;
    profile.cache_misses = 100;

    assert!((profile.hit_rate() - 0.9).abs() < 0.01);
}

#[test]
fn test_memory_profile_naive_gemm() {
    let profile = MemoryProfile::profile_naive_gemm(64, 64, 64, 64);

    assert!(profile.total_accesses > 0);
    assert!(profile.sequential_accesses > 0);
    assert!(profile.strided_accesses > 0);
    assert!(profile.cache_hits + profile.cache_misses <= profile.total_accesses);
}

#[test]
fn test_memory_profile_tiled_gemm() {
    let profile = MemoryProfile::profile_tiled_gemm(64, 64, 64, 16, 64);

    assert!(profile.total_accesses > 0);
    assert!(profile.sequential_accesses > 0);
    assert!(profile.cache_hits + profile.cache_misses <= profile.total_accesses);
}

#[test]
fn test_tiled_vs_naive_cache_behavior() {
    let naive = MemoryProfile::profile_naive_gemm(256, 256, 256, 64);
    let tiled = MemoryProfile::profile_tiled_gemm(256, 256, 256, 32, 64);

    // Tiled should have better cache hit rate
    assert!(tiled.hit_rate() >= naive.hit_rate());
}

#[test]
fn test_various_matrix_sizes() {
    let sizes = vec![32, 64, 128, 256];

    for size in sizes {
        let profile = MemoryProfile::profile_naive_gemm(size, size, size, 64);
        assert!(profile.total_accesses > 0);
        assert!(profile.hit_rate() >= 0.0 && profile.hit_rate() <= 1.0);
    }
}
