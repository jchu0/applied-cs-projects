//! Integration tests for the GPU GEMM optimization library.

use gpu_gemm_optimization::{
    Matrix, naive_gemm, tiled_gemm, register_tiled_gemm, parallel_tiled_gemm, GemmConfig, GemmMetrics,
};
use gpu_gemm_optimization::gemm::{double_buffered_gemm, scaled_gemm};
use gpu_gemm_optimization::vectorize::{vectorized_gemm, vectorized_gemm_transposed_b, CoalescingAnalysis};
use gpu_gemm_optimization::memory::{BankConflictAnalysis, SharedMemoryConfig, OccupancyCalculator, KernelRequirements};
use gpu_gemm_optimization::metrics::{Benchmark, PerformanceModel};
use gpu_gemm_optimization::autotuner::{Autotuner, AutotuneConfig, ParameterSpace, SearchStrategy};
use std::time::Instant;

/// Helper to verify GEMM result against naive computation.
fn verify_gemm(a: &Matrix, b: &Matrix, c: &Matrix, tolerance: f32) -> bool {
    let mut expected = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            expected.set(i, j, sum);
        }
    }
    c.approx_eq(&expected, tolerance)
}

#[test]
fn test_end_to_end_gemm_workflow() {
    // Step 1: Create test matrices
    let m = 128;
    let n = 128;
    let k = 128;

    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);
    let mut c = Matrix::zeros(m, n);

    // Step 2: Run baseline naive GEMM
    let start = Instant::now();
    naive_gemm(&a, &b, &mut c).unwrap();
    let baseline_time = start.elapsed();

    // Verify result
    assert!(verify_gemm(&a, &b, &c, 1e-3));

    // Step 3: Run optimized tiled GEMM
    let mut c_tiled = Matrix::zeros(m, n);
    let start = Instant::now();
    tiled_gemm(&a, &b, &mut c_tiled, 16).unwrap();
    let tiled_time = start.elapsed();

    // Verify correctness
    assert!(c_tiled.approx_eq(&c, 1e-3), "Tiled result doesn't match naive");

    // Step 4: Run parallel GEMM
    let mut c_parallel = Matrix::zeros(m, n);
    let start = Instant::now();
    parallel_tiled_gemm(&a, &b, &mut c_parallel, 16).unwrap();
    let parallel_time = start.elapsed();

    // Verify correctness
    assert!(c_parallel.approx_eq(&c, 1e-3), "Parallel result doesn't match naive");

    // Print performance comparison
    println!("GEMM {}x{}x{} Performance:", m, n, k);
    println!("  Naive:    {:?}", baseline_time);
    println!("  Tiled:    {:?}", tiled_time);
    println!("  Parallel: {:?}", parallel_time);
}

#[test]
fn test_various_matrix_sizes() {
    let test_cases = vec![
        (64, 64, 64),     // Small
        (128, 256, 128),  // Rectangular
        (256, 256, 256),  // Medium square
        (100, 200, 150),  // Odd sizes
    ];

    for (m, n, k) in test_cases {
        let a = Matrix::random(m, k);
        let b = Matrix::random(k, n);
        let mut c = Matrix::zeros(m, n);

        naive_gemm(&a, &b, &mut c).unwrap();

        // Result should be finite and reasonable
        let norm = c.frobenius_norm();
        assert!(norm.is_finite(), "Result not finite for {}x{}x{}", m, n, k);
        assert!(norm > 0.0, "Result is zero for {}x{}x{}", m, n, k);
    }
}

#[test]
fn test_all_kernels_produce_same_result() {
    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let mut c_naive = Matrix::zeros(64, 64);
    let mut c_tiled = Matrix::zeros(64, 64);
    let mut c_register = Matrix::zeros(64, 64);
    let mut c_parallel = Matrix::zeros(64, 64);
    let mut c_double = Matrix::zeros(64, 64);

    naive_gemm(&a, &b, &mut c_naive).unwrap();
    tiled_gemm(&a, &b, &mut c_tiled, 16).unwrap();
    register_tiled_gemm(&a, &b, &mut c_register, &GemmConfig::default()).unwrap();
    parallel_tiled_gemm(&a, &b, &mut c_parallel, 16).unwrap();
    double_buffered_gemm(&a, &b, &mut c_double, 16).unwrap();

    assert!(c_tiled.approx_eq(&c_naive, 1e-3), "Tiled mismatch");
    assert!(c_register.approx_eq(&c_naive, 1e-3), "Register mismatch");
    assert!(c_parallel.approx_eq(&c_naive, 1e-3), "Parallel mismatch");
    assert!(c_double.approx_eq(&c_naive, 1e-3), "Double-buffered mismatch");
}

#[test]
fn test_scaled_gemm_integration() {
    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);
    let mut c = Matrix::ones(64, 64);
    let c_orig = c.clone();

    // C = 2.0 * A*B + 0.5 * C
    scaled_gemm(&a, &b, &mut c, 2.0, 0.5).unwrap();

    // Verify manually
    let mut ab = Matrix::zeros(64, 64);
    naive_gemm(&a, &b, &mut ab).unwrap();

    for i in 0..64 {
        for j in 0..64 {
            let expected = 2.0 * ab.get(i, j) + 0.5 * c_orig.get(i, j);
            assert!((c.get(i, j) - expected).abs() < 1e-3);
        }
    }
}

#[test]
fn test_vectorized_gemm_integration() {
    // Create test data for vectorized GEMM
    let m = 64;
    let n = 64;
    let k = 64;

    let a: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01) % 1.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i + 100) as f32 * 0.01) % 1.0).collect();
    let mut c = vec![0.0f32; m * n];

    vectorized_gemm(&a, &b, &mut c, m, n, k);

    // Verify result is non-zero and finite
    let sum: f32 = c.iter().sum();
    assert!(sum.is_finite());
    assert!(sum > 0.0);
}

#[test]
fn test_vectorized_gemm_transposed() {
    let m = 64;
    let n = 64;
    let k = 64;

    let a: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01) % 1.0).collect();
    let b_t: Vec<f32> = (0..n*k).map(|i| ((i + 50) as f32 * 0.01) % 1.0).collect();
    let mut c = vec![0.0f32; m * n];

    vectorized_gemm_transposed_b(&a, &b_t, &mut c, m, n, k);

    let sum: f32 = c.iter().sum();
    assert!(sum.is_finite());
    assert!(sum > 0.0);
}

#[test]
fn test_coalescing_analysis_integration() {
    // Test row-major sequential access (should be coalesced)
    let sequential: Vec<(usize, usize)> = (0..16).map(|i| (0, i)).collect();
    let analysis = CoalescingAnalysis::analyze(&sequential, true, 16);

    assert!(analysis.efficiency > 0.9);
    assert_eq!(analysis.strided_count, 0);

    // Test column-major access in row-major layout (strided)
    let strided: Vec<(usize, usize)> = (0..16).map(|i| (i, 0)).collect();
    let strided_analysis = CoalescingAnalysis::analyze(&strided, true, 16);

    assert!(strided_analysis.efficiency < 0.5);
}

#[test]
fn test_bank_conflict_analysis_integration() {
    // Perfect sequential access (no conflicts) - each access to different bank
    let sequential: Vec<usize> = (0..32).map(|i| i * 4).collect();  // 4-byte stride = 1 bank
    let analysis = BankConflictAnalysis::analyze(&sequential);

    assert_eq!(analysis.conflict_count, 0);

    // Strided access that causes conflicts - all access same bank
    let strided: Vec<usize> = (0..8).map(|i| i * 32 * 4).collect();  // Stride of 32 banks
    let strided_analysis = BankConflictAnalysis::analyze(&strided);

    assert!(strided_analysis.conflict_count > 0);
}

#[test]
fn test_shared_memory_config_integration() {
    // Create shared memory config for a 32x32 tile of f32 elements
    let config = SharedMemoryConfig::new(32, 32, 4);

    assert_eq!(config.stride(), 32);
    assert_eq!(config.index(1, 0), 32);
    assert!(config.size_bytes() > 0);
    assert!(config.fits_in_shared_memory());

    // Test with auto-padding
    let padded_config = SharedMemoryConfig::with_auto_padding(32, 32, 4);
    assert!(padded_config.stride() >= 32);
}

#[test]
fn test_occupancy_calculator_integration() {
    let calc = OccupancyCalculator::ampere();

    let reqs = KernelRequirements {
        threads_per_block: 256,
        registers_per_thread: 32,
        shared_mem_per_block: 16384,
    };

    let occ = calc.calculate(&reqs);

    assert!(occ.occupancy >= 0.0 && occ.occupancy <= 100.0);
    assert!(occ.warps_per_sm > 0);
}

#[test]
fn test_performance_model_integration() {
    let model = PerformanceModel::default();

    // Test roofline prediction
    let small_ai = PerformanceModel::gemm_arithmetic_intensity(32, 32, 32);
    let large_ai = PerformanceModel::gemm_arithmetic_intensity(1024, 1024, 1024);

    // Large GEMM should have higher arithmetic intensity
    assert!(large_ai > small_ai);

    // Predictions should be reasonable
    let small_pred = model.predict_gflops(small_ai);
    let large_pred = model.predict_gflops(large_ai);

    assert!(small_pred > 0.0);
    assert!(large_pred > 0.0);
}

#[test]
fn test_benchmark_integration() {
    let bench = Benchmark::new(2, 5, 100.0);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let metrics = bench.run(&a, &b, |a, b, c| naive_gemm(a, b, c)).unwrap();

    assert!(metrics.gflops > 0.0);
    assert!(metrics.bandwidth_gbs > 0.0);
    assert!(metrics.arithmetic_intensity > 0.0);
}

#[test]
fn test_autotuner_integration() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::GridSearch,
        max_trials: 4,
        time_budget: None,
        early_stop: 10,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace {
        block_m: vec![32, 64],
        block_n: vec![32, 64],
        block_k: vec![8],
        thread_m: vec![8],
        thread_n: vec![8],
    };

    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    }).unwrap();

    assert!(result.num_trials > 0);
    assert!(result.best_metrics.gflops > 0.0);
    assert!(result.best_config.validate().is_ok());
}

#[test]
fn test_full_optimization_workflow() {
    let m = 128;
    let n = 128;
    let k = 128;

    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);

    // 1. Analyze memory access pattern
    let pattern: Vec<(usize, usize)> = (0..16).map(|i| (0, i)).collect();
    let coalescing = CoalescingAnalysis::analyze(&pattern, true, k);
    assert!(coalescing.efficiency > 0.5);

    // 2. Configure shared memory for tiles
    let smem_a = SharedMemoryConfig::with_auto_padding(16, 16, 4);
    let smem_b = SharedMemoryConfig::with_auto_padding(16, 16, 4);
    let total_smem = smem_a.size_bytes() + smem_b.size_bytes();

    // 3. Calculate occupancy
    let calc = OccupancyCalculator::ampere();
    let reqs = KernelRequirements {
        threads_per_block: 256,
        registers_per_thread: 32,
        shared_mem_per_block: total_smem,
    };
    let occ = calc.calculate(&reqs);
    assert!(occ.occupancy > 0.0);

    // 4. Run optimized kernel
    let mut c = Matrix::zeros(m, n);
    let config = GemmConfig {
        block_m: 64,
        block_n: 64,
        block_k: 8,
        thread_m: 8,
        thread_n: 8,
    };
    register_tiled_gemm(&a, &b, &mut c, &config).unwrap();

    // 5. Verify result
    let mut c_ref = Matrix::zeros(m, n);
    naive_gemm(&a, &b, &mut c_ref).unwrap();
    assert!(c.approx_eq(&c_ref, 1e-3));
}

#[test]
fn test_numerical_stability() {
    let scales = vec![1e-6, 1e-3, 1.0, 1e3, 1e6];

    for scale in scales {
        let mut a = Matrix::random(64, 64);
        let mut b = Matrix::random(64, 64);

        // Scale A up and B down to keep product reasonable
        for i in 0..64 {
            for j in 0..64 {
                a.set(i, j, a.get(i, j) * scale);
                b.set(i, j, b.get(i, j) / scale);
            }
        }

        let mut c = Matrix::zeros(64, 64);
        naive_gemm(&a, &b, &mut c).unwrap();

        let norm = c.frobenius_norm();
        assert!(norm.is_finite(), "Not finite for scale {}", scale);
        assert!(norm > 0.0, "Zero result for scale {}", scale);
    }
}

#[test]
fn test_identity_matrix_gemm() {
    let size = 64;
    let identity = Matrix::identity(size);
    let random = Matrix::random(size, size);
    let mut result = Matrix::zeros(size, size);

    // I * A = A
    naive_gemm(&identity, &random, &mut result).unwrap();
    assert!(result.approx_eq(&random, 1e-6));

    // A * I = A
    naive_gemm(&random, &identity, &mut result).unwrap();
    assert!(result.approx_eq(&random, 1e-6));
}

#[test]
fn test_zero_matrix_gemm() {
    let size = 64;
    let zeros = Matrix::zeros(size, size);
    let random = Matrix::random(size, size);
    let mut result = Matrix::ones(size, size);

    naive_gemm(&zeros, &random, &mut result).unwrap();

    // Result should be all zeros
    for i in 0..size {
        for j in 0..size {
            assert_eq!(result.get(i, j), 0.0);
        }
    }
}
