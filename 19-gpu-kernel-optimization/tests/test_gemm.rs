//! Tests for GEMM kernel implementations.

use gpu_gemm_optimization::{
    Matrix, naive_gemm, tiled_gemm, register_tiled_gemm, parallel_tiled_gemm, GemmConfig,
};
use gpu_gemm_optimization::gemm::{double_buffered_gemm, scaled_gemm};

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
fn test_naive_gemm_basic() {
    let a = Matrix::random(4, 6);
    let b = Matrix::random(6, 5);
    let mut c = Matrix::zeros(4, 5);

    naive_gemm(&a, &b, &mut c).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-4));
}

#[test]
fn test_naive_gemm_square_matrices() {
    let sizes = vec![1, 2, 4, 8, 16, 32, 64];

    for size in sizes {
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);
        let mut c = Matrix::zeros(size, size);

        naive_gemm(&a, &b, &mut c).unwrap();

        assert!(
            verify_gemm(&a, &b, &c, 1e-3),
            "Failed for size {}x{}",
            size,
            size
        );
    }
}

#[test]
fn test_naive_gemm_rectangular() {
    let test_cases = vec![
        (10, 20, 30),
        (20, 10, 30),
        (30, 30, 10),
        (1, 100, 50),
        (100, 1, 50),
        (100, 50, 1),
    ];

    for (m, n, k) in test_cases {
        let a = Matrix::random(m, k);
        let b = Matrix::random(k, n);
        let mut c = Matrix::zeros(m, n);

        naive_gemm(&a, &b, &mut c).unwrap();

        assert!(
            verify_gemm(&a, &b, &c, 1e-3),
            "Failed for dimensions {}x{}x{}",
            m, n, k
        );
    }
}

#[test]
fn test_naive_gemm_identity() {
    let size = 10;
    let a = Matrix::identity(size);
    let b = Matrix::random(size, size);
    let mut c = Matrix::zeros(size, size);

    naive_gemm(&a, &b, &mut c).unwrap();

    // Result should be equal to b
    assert!(c.approx_eq(&b, 1e-6));
}

#[test]
fn test_naive_gemm_zeros() {
    let a = Matrix::zeros(5, 5);
    let b = Matrix::random(5, 5);
    let mut c = Matrix::ones(5, 5);

    naive_gemm(&a, &b, &mut c).unwrap();

    // Result should be all zeros
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(c.get(i, j), 0.0);
        }
    }
}

#[test]
fn test_naive_gemm_dimension_mismatch() {
    let a = Matrix::random(4, 7); // Wrong k dimension
    let b = Matrix::random(6, 5);
    let mut c = Matrix::zeros(4, 5);

    let result = naive_gemm(&a, &b, &mut c);
    assert!(result.is_err());
}

#[test]
fn test_tiled_gemm_basic() {
    let a = Matrix::random(32, 32);
    let b = Matrix::random(32, 32);
    let mut c = Matrix::zeros(32, 32);

    tiled_gemm(&a, &b, &mut c, 8).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_tiled_gemm_various_tile_sizes() {
    let tile_sizes = vec![4, 8, 16, 32];
    let size = 64;

    for tile_size in tile_sizes {
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);
        let mut c = Matrix::zeros(size, size);

        tiled_gemm(&a, &b, &mut c, tile_size).unwrap();

        assert!(
            verify_gemm(&a, &b, &c, 1e-3),
            "Failed for tile_size {}",
            tile_size
        );
    }
}

#[test]
fn test_tiled_gemm_non_divisible() {
    // Size not divisible by tile size
    let a = Matrix::random(33, 33);
    let b = Matrix::random(33, 33);
    let mut c = Matrix::zeros(33, 33);

    tiled_gemm(&a, &b, &mut c, 8).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_register_tiled_gemm_basic() {
    let config = GemmConfig::default();
    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);
    let mut c = Matrix::zeros(64, 64);

    register_tiled_gemm(&a, &b, &mut c, &config).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_register_tiled_gemm_custom_config() {
    let config = GemmConfig {
        block_m: 32,
        block_n: 32,
        block_k: 8,
        thread_m: 8,
        thread_n: 8,
    };

    let a = Matrix::random(128, 128);
    let b = Matrix::random(128, 128);
    let mut c = Matrix::zeros(128, 128);

    register_tiled_gemm(&a, &b, &mut c, &config).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_parallel_tiled_gemm_basic() {
    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);
    let mut c = Matrix::zeros(64, 64);

    parallel_tiled_gemm(&a, &b, &mut c, 16).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_parallel_tiled_gemm_large() {
    let a = Matrix::random(256, 256);
    let b = Matrix::random(256, 256);
    let mut c = Matrix::zeros(256, 256);

    parallel_tiled_gemm(&a, &b, &mut c, 32).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-2));
}

#[test]
fn test_double_buffered_gemm_basic() {
    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);
    let mut c = Matrix::zeros(64, 64);

    double_buffered_gemm(&a, &b, &mut c, 16).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_double_buffered_gemm_rectangular() {
    let a = Matrix::random(48, 64);
    let b = Matrix::random(64, 32);
    let mut c = Matrix::zeros(48, 32);

    double_buffered_gemm(&a, &b, &mut c, 16).unwrap();

    assert!(verify_gemm(&a, &b, &c, 1e-3));
}

#[test]
fn test_scaled_gemm_alpha() {
    let a = Matrix::random(16, 16);
    let b = Matrix::random(16, 16);
    let mut c = Matrix::zeros(16, 16);

    scaled_gemm(&a, &b, &mut c, 2.0, 0.0).unwrap();

    // Verify scaling by 2
    let mut expected = Matrix::zeros(16, 16);
    naive_gemm(&a, &b, &mut expected).unwrap();

    for i in 0..16 {
        for j in 0..16 {
            assert!((c.get(i, j) - 2.0 * expected.get(i, j)).abs() < 1e-3);
        }
    }
}

#[test]
fn test_scaled_gemm_beta() {
    let a = Matrix::random(16, 16);
    let b = Matrix::random(16, 16);
    let mut c = Matrix::ones(16, 16);
    let c_orig = c.clone();

    // C = 1.0 * A*B + 0.5 * C_orig
    scaled_gemm(&a, &b, &mut c, 1.0, 0.5).unwrap();

    // Verify: C = A*B + 0.5 * ones
    let mut ab = Matrix::zeros(16, 16);
    naive_gemm(&a, &b, &mut ab).unwrap();

    for i in 0..16 {
        for j in 0..16 {
            let expected = ab.get(i, j) + 0.5 * c_orig.get(i, j);
            assert!((c.get(i, j) - expected).abs() < 1e-3);
        }
    }
}

#[test]
fn test_gemm_config_default() {
    let config = GemmConfig::default();
    assert_eq!(config.block_m, 64);
    assert_eq!(config.block_n, 64);
    assert_eq!(config.block_k, 8);
    assert_eq!(config.thread_m, 8);
    assert_eq!(config.thread_n, 8);
}

#[test]
fn test_gemm_config_validate_success() {
    let config = GemmConfig {
        block_m: 64,
        block_n: 64,
        block_k: 8,
        thread_m: 8,
        thread_n: 8,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_gemm_config_validate_failure() {
    let config = GemmConfig {
        block_m: 64,
        block_n: 64,
        block_k: 8,
        thread_m: 7, // Not divisible
        thread_n: 8,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_all_kernels_match() {
    // Verify all kernel implementations produce the same result
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

    // All should match naive
    assert!(c_tiled.approx_eq(&c_naive, 1e-3), "Tiled GEMM mismatch");
    assert!(c_register.approx_eq(&c_naive, 1e-3), "Register GEMM mismatch");
    assert!(c_parallel.approx_eq(&c_naive, 1e-3), "Parallel GEMM mismatch");
    assert!(c_double.approx_eq(&c_naive, 1e-3), "Double-buffered GEMM mismatch");
}
