//! Tests for the Matrix module.

use gpu_gemm_optimization::Matrix;
use gpu_gemm_optimization::matrix::{Layout, convert_layout, pack_matrix_a, pack_matrix_b};

#[test]
fn test_matrix_zeros() {
    let mat = Matrix::zeros(3, 4);
    assert_eq!(mat.rows, 3);
    assert_eq!(mat.cols, 4);
    assert_eq!(mat.data.len(), 12);
    for &val in &mat.data {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_matrix_ones() {
    let mat = Matrix::ones(2, 5);
    assert_eq!(mat.rows, 2);
    assert_eq!(mat.cols, 5);
    for &val in &mat.data {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_matrix_random() {
    let mat = Matrix::random(10, 20);
    assert_eq!(mat.rows, 10);
    assert_eq!(mat.cols, 20);

    // Check that values are within expected range
    for &val in &mat.data {
        assert!(val >= -1.0 && val <= 1.0);
    }
}

#[test]
fn test_matrix_identity() {
    let mat = Matrix::identity(5);
    assert_eq!(mat.rows, 5);
    assert_eq!(mat.cols, 5);

    for i in 0..5 {
        for j in 0..5 {
            if i == j {
                assert_eq!(mat.get(i, j), 1.0);
            } else {
                assert_eq!(mat.get(i, j), 0.0);
            }
        }
    }
}

#[test]
fn test_matrix_get_set() {
    let mut mat = Matrix::zeros(3, 3);

    mat.set(1, 2, 5.0);
    assert_eq!(mat.get(1, 2), 5.0);

    mat.set(0, 0, -3.0);
    assert_eq!(mat.get(0, 0), -3.0);
}

#[test]
fn test_matrix_get_mut() {
    let mut mat = Matrix::zeros(2, 2);
    *mat.get_mut(0, 1) = 42.0;
    assert_eq!(mat.get(0, 1), 42.0);
}

#[test]
fn test_matrix_row_ptr() {
    let mut mat = Matrix::zeros(3, 4);
    for i in 0..3 {
        for j in 0..4 {
            mat.set(i, j, (i * 4 + j) as f32);
        }
    }

    let row1 = mat.row_ptr(1);
    assert_eq!(row1.len(), 4);
    assert_eq!(row1, &[4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_matrix_transpose() {
    let mut mat = Matrix::zeros(2, 3);
    // [1, 2, 3]
    // [4, 5, 6]
    mat.set(0, 0, 1.0); mat.set(0, 1, 2.0); mat.set(0, 2, 3.0);
    mat.set(1, 0, 4.0); mat.set(1, 1, 5.0); mat.set(1, 2, 6.0);

    let transposed = mat.transpose();

    assert_eq!(transposed.rows, 3);
    assert_eq!(transposed.cols, 2);

    // [1, 4]
    // [2, 5]
    // [3, 6]
    assert_eq!(transposed.get(0, 0), 1.0);
    assert_eq!(transposed.get(1, 0), 2.0);
    assert_eq!(transposed.get(2, 0), 3.0);
    assert_eq!(transposed.get(0, 1), 4.0);
    assert_eq!(transposed.get(1, 1), 5.0);
    assert_eq!(transposed.get(2, 1), 6.0);
}

#[test]
fn test_matrix_approx_eq() {
    let a = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        cols: 2,
    };
    let b = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        cols: 2,
    };
    let c = Matrix {
        data: vec![1.0, 2.0, 3.0, 5.0],
        rows: 2,
        cols: 2,
    };

    assert!(a.approx_eq(&b, 1e-10));
    assert!(!a.approx_eq(&c, 1e-10));
}

#[test]
fn test_matrix_approx_eq_tolerance() {
    let a = Matrix {
        data: vec![1.0, 2.0],
        rows: 1,
        cols: 2,
    };
    let b = Matrix {
        data: vec![1.0001, 2.0001],
        rows: 1,
        cols: 2,
    };

    assert!(a.approx_eq(&b, 1e-3));
    assert!(!a.approx_eq(&b, 1e-5));
}

#[test]
fn test_matrix_max_diff() {
    let a = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        cols: 2,
    };
    let b = Matrix {
        data: vec![1.5, 2.0, 3.0, 4.0],
        rows: 2,
        cols: 2,
    };

    assert!((a.max_diff(&b) - 0.5).abs() < 1e-6);
}

#[test]
fn test_matrix_frobenius_norm() {
    let mat = Matrix {
        data: vec![3.0, 4.0, 0.0, 5.0],
        rows: 2,
        cols: 2,
    };
    let norm = mat.frobenius_norm();
    let expected = (9.0 + 16.0 + 0.0 + 25.0_f32).sqrt();
    assert!((norm - expected).abs() < 1e-6);
}

#[test]
fn test_layout_convert_row_to_col() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Row major 2x3:
    // [1, 2, 3]
    // [4, 5, 6]

    let col_major = convert_layout(&data, 2, 3, Layout::RowMajor, Layout::ColumnMajor);
    // Col major: [1, 4, 2, 5, 3, 6]
    assert_eq!(col_major, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_layout_convert_col_to_row() {
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    // Col major 2x3

    let row_major = convert_layout(&data, 2, 3, Layout::ColumnMajor, Layout::RowMajor);
    // Row major: [1, 2, 3, 4, 5, 6]
    assert_eq!(row_major, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_layout_same_layout() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = convert_layout(&data, 2, 2, Layout::RowMajor, Layout::RowMajor);
    assert_eq!(result, data);
}

#[test]
fn test_pack_matrix_a() {
    let mat = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        rows: 3,
        cols: 3,
    };

    // Pack with block size 2x2
    let packed = pack_matrix_a(&mat, 2, 2);

    // Should have 2x2 blocks: ceil(3/2)=2 blocks in each dim
    // Total size: 2 * 2 * 2 * 2 = 16
    assert_eq!(packed.len(), 16);

    // First block (0,0): [1,2,4,5]
    assert_eq!(packed[0], 1.0);
    assert_eq!(packed[1], 2.0);
    assert_eq!(packed[2], 4.0);
    assert_eq!(packed[3], 5.0);
}

#[test]
fn test_pack_matrix_b() {
    let mat = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        rows: 3,
        cols: 3,
    };

    let packed = pack_matrix_b(&mat, 2, 2);
    assert_eq!(packed.len(), 16);
}

#[test]
fn test_large_matrix_operations() {
    let size = 256;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);

    // Just verify operations don't panic on larger sizes
    let _at = a.transpose();
    let _norm = b.frobenius_norm();
    let _diff = a.max_diff(&b);
}

#[test]
fn test_matrix_numerical_stability() {
    let mut mat = Matrix::zeros(10, 10);

    for i in 0..10 {
        for j in 0..10 {
            let val = if (i + j) % 2 == 0 { 1e-10 } else { 1e10 };
            mat.set(i, j, val);
        }
    }

    let norm = mat.frobenius_norm();
    assert!(norm.is_finite());
    assert!(norm > 0.0);
}

#[test]
fn test_matrix_clone() {
    let a = Matrix::random(5, 5);
    let b = a.clone();

    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.data, b.data);
}
