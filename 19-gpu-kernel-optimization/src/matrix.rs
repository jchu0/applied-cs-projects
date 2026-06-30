//! Matrix data structure and utilities.

use crate::{Error, Result};
use rand::Rng;

/// A dense matrix stored in row-major order.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Matrix data in row-major order.
    pub data: Vec<f32>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Matrix {
    /// Create a new matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a new matrix filled with ones.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a matrix with random values.
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Self { data, rows, cols }
    }

    /// Create an identity matrix.
    pub fn identity(size: usize) -> Self {
        let mut m = Self::zeros(size, size);
        for i in 0..size {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Get mutable reference to element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f32 {
        &mut self.data[row * self.cols + col]
    }

    /// Get pointer to start of row.
    #[inline]
    pub fn row_ptr(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get mutable pointer to start of row.
    #[inline]
    pub fn row_ptr_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Compare with another matrix within tolerance.
    pub fn approx_eq(&self, other: &Matrix, tolerance: f32) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.data.len() {
            let diff = (self.data[i] - other.data[i]).abs();
            let max_val = self.data[i].abs().max(other.data[i].abs()).max(1.0);
            if diff / max_val > tolerance {
                return false;
            }
        }
        true
    }

    /// Get maximum absolute difference from another matrix.
    pub fn max_diff(&self, other: &Matrix) -> f32 {
        if self.rows != other.rows || self.cols != other.cols {
            return f32::INFINITY;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// Get maximum relative error from another matrix.
    pub fn max_relative_error(&self, other: &Matrix) -> f32 {
        if self.rows != other.rows || self.cols != other.cols {
            return f32::INFINITY;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let diff = (a - b).abs();
                let max_val = a.abs().max(b.abs()).max(1e-8);
                diff / max_val
            })
            .fold(0.0, f32::max)
    }

    /// Calculate Frobenius norm.
    pub fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Memory layout for matrix storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major (C style): A[i][j] = A[i*cols + j]
    RowMajor,
    /// Column-major (Fortran style): A[i][j] = A[j*rows + i]
    ColumnMajor,
}

/// Convert between layouts.
pub fn convert_layout(data: &[f32], rows: usize, cols: usize, from: Layout, to: Layout) -> Vec<f32> {
    if from == to {
        return data.to_vec();
    }

    let mut result = vec![0.0; rows * cols];

    match (from, to) {
        (Layout::RowMajor, Layout::ColumnMajor) => {
            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = data[i * cols + j];
                }
            }
        }
        (Layout::ColumnMajor, Layout::RowMajor) => {
            for i in 0..rows {
                for j in 0..cols {
                    result[i * cols + j] = data[j * rows + i];
                }
            }
        }
        _ => unreachable!(),
    }

    result
}

/// Pack matrix data for efficient access.
///
/// This simulates the GPU technique of packing data for coalesced access.
pub fn pack_matrix_a(
    a: &Matrix,
    block_m: usize,
    block_k: usize,
) -> Vec<f32> {
    let m = a.rows;
    let k = a.cols;
    let num_blocks_m = (m + block_m - 1) / block_m;
    let num_blocks_k = (k + block_k - 1) / block_k;

    let mut packed = vec![0.0; num_blocks_m * num_blocks_k * block_m * block_k];

    for bm in 0..num_blocks_m {
        for bk in 0..num_blocks_k {
            let block_offset = (bm * num_blocks_k + bk) * block_m * block_k;

            for i in 0..block_m {
                for j in 0..block_k {
                    let row = bm * block_m + i;
                    let col = bk * block_k + j;

                    let value = if row < m && col < k {
                        a.get(row, col)
                    } else {
                        0.0
                    };

                    packed[block_offset + i * block_k + j] = value;
                }
            }
        }
    }

    packed
}

/// Pack matrix B for efficient access.
pub fn pack_matrix_b(
    b: &Matrix,
    block_k: usize,
    block_n: usize,
) -> Vec<f32> {
    let k = b.rows;
    let n = b.cols;
    let num_blocks_k = (k + block_k - 1) / block_k;
    let num_blocks_n = (n + block_n - 1) / block_n;

    let mut packed = vec![0.0; num_blocks_k * num_blocks_n * block_k * block_n];

    for bk in 0..num_blocks_k {
        for bn in 0..num_blocks_n {
            let block_offset = (bk * num_blocks_n + bn) * block_k * block_n;

            for i in 0..block_k {
                for j in 0..block_n {
                    let row = bk * block_k + i;
                    let col = bn * block_n + j;

                    let value = if row < k && col < n {
                        b.get(row, col)
                    } else {
                        0.0
                    };

                    packed[block_offset + i * block_n + j] = value;
                }
            }
        }
    }

    packed
}
