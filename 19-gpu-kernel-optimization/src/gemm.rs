//! GEMM kernel implementations with progressive optimizations.

use crate::matrix::Matrix;
use crate::{Error, Result};
use rayon::prelude::*;

/// GEMM configuration parameters (simulating GPU thread block config).
#[derive(Debug, Clone, Copy)]
pub struct GemmConfig {
    /// Block tile size for M dimension.
    pub block_m: usize,
    /// Block tile size for N dimension.
    pub block_n: usize,
    /// Block tile size for K dimension.
    pub block_k: usize,
    /// Thread tile size for M dimension.
    pub thread_m: usize,
    /// Thread tile size for N dimension.
    pub thread_n: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            block_m: 64,
            block_n: 64,
            block_k: 8,
            thread_m: 8,
            thread_n: 8,
        }
    }
}

impl GemmConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.block_m % self.thread_m != 0 {
            return Err(Error::InvalidConfig(
                "block_m must be divisible by thread_m".into(),
            ));
        }
        if self.block_n % self.thread_n != 0 {
            return Err(Error::InvalidConfig(
                "block_n must be divisible by thread_n".into(),
            ));
        }
        Ok(())
    }
}

/// Trait for GEMM kernel implementations.
pub trait GemmKernel: Send + Sync {
    /// Compute C = A * B
    fn compute(&self, a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<()>;

    /// Get kernel name.
    fn name(&self) -> &'static str;
}

/// Stage 1: Naive GEMM implementation.
///
/// One "thread" per output element, no data reuse.
/// This mirrors the GPU naive implementation.
pub fn naive_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(Error::DimensionMismatch(format!(
            "C dimensions ({}, {}) don't match result ({}, {})",
            c.rows, c.cols, a.rows, b.cols
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // One computation per output element (like GPU thread per element)
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a.get(row, i) * b.get(i, col);
            }
            c.set(row, col, sum);
        }
    }

    Ok(())
}

/// Stage 2: Tiled GEMM with simulated shared memory.
///
/// This demonstrates block-level tiling where tiles are loaded into
/// "shared memory" (local arrays) for reuse.
pub fn tiled_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, tile_size: usize) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // Iterate over output tiles (like thread blocks)
    for tile_row in (0..m).step_by(tile_size) {
        for tile_col in (0..n).step_by(tile_size) {
            // "Shared memory" tiles
            let mut a_tile = vec![0.0f32; tile_size * tile_size];
            let mut b_tile = vec![0.0f32; tile_size * tile_size];

            // Accumulator (simulating registers)
            let mut c_acc = vec![0.0f32; tile_size * tile_size];

            // Iterate over K dimension tiles
            for tile_k in (0..k).step_by(tile_size) {
                // Load A tile into "shared memory"
                for i in 0..tile_size {
                    for j in 0..tile_size {
                        let row = tile_row + i;
                        let col = tile_k + j;
                        a_tile[i * tile_size + j] = if row < m && col < k {
                            a.get(row, col)
                        } else {
                            0.0
                        };
                    }
                }

                // Load B tile into "shared memory"
                for i in 0..tile_size {
                    for j in 0..tile_size {
                        let row = tile_k + i;
                        let col = tile_col + j;
                        b_tile[i * tile_size + j] = if row < k && col < n {
                            b.get(row, col)
                        } else {
                            0.0
                        };
                    }
                }

                // Compute on tiles (simulating __syncthreads and computation)
                for i in 0..tile_size {
                    for j in 0..tile_size {
                        let mut sum = 0.0f32;
                        for kk in 0..tile_size {
                            sum += a_tile[i * tile_size + kk] * b_tile[kk * tile_size + j];
                        }
                        c_acc[i * tile_size + j] += sum;
                    }
                }
            }

            // Write results to C
            for i in 0..tile_size {
                for j in 0..tile_size {
                    let row = tile_row + i;
                    let col = tile_col + j;
                    if row < m && col < n {
                        c.set(row, col, c_acc[i * tile_size + j]);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Stage 3: Register tiled GEMM.
///
/// Each "thread" computes a TM x TN tile using registers for accumulation.
/// This demonstrates the register blocking optimization.
pub fn register_tiled_gemm(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    config: &GemmConfig,
) -> Result<()> {
    config.validate()?;

    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    let bm = config.block_m;
    let bn = config.block_n;
    let bk = config.block_k;
    let tm = config.thread_m;
    let tn = config.thread_n;

    // Iterate over block tiles
    for block_row in (0..m).step_by(bm) {
        for block_col in (0..n).step_by(bn) {
            // Number of "threads" per block
            let threads_m = bm / tm;
            let threads_n = bn / tn;

            // Each "thread" processes a TM x TN tile
            for thread_y in 0..threads_m {
                for thread_x in 0..threads_n {
                    // Register accumulators for this thread
                    let mut c_regs = [[0.0f32; 8]; 8]; // TM x TN
                    let mut a_regs = [0.0f32; 8]; // TM
                    let mut b_regs = [0.0f32; 8]; // TN

                    // Thread's output position
                    let row_start = block_row + thread_y * tm;
                    let col_start = block_col + thread_x * tn;

                    // Iterate over K in blocks
                    for kk in (0..k).step_by(bk) {
                        // Load A and B tiles (simulating shared memory load)
                        // Then compute outer product

                        for k_offset in 0..bk.min(k - kk) {
                            // Load A column into registers
                            for i in 0..tm {
                                let row = row_start + i;
                                let col = kk + k_offset;
                                a_regs[i] = if row < m && col < k {
                                    a.get(row, col)
                                } else {
                                    0.0
                                };
                            }

                            // Load B row into registers
                            for j in 0..tn {
                                let row = kk + k_offset;
                                let col = col_start + j;
                                b_regs[j] = if row < k && col < n {
                                    b.get(row, col)
                                } else {
                                    0.0
                                };
                            }

                            // Outer product accumulation
                            for i in 0..tm {
                                for j in 0..tn {
                                    c_regs[i][j] += a_regs[i] * b_regs[j];
                                }
                            }
                        }
                    }

                    // Store results
                    for i in 0..tm {
                        for j in 0..tn {
                            let row = row_start + i;
                            let col = col_start + j;
                            if row < m && col < n {
                                c.set(row, col, c_regs[i][j]);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Parallel tiled GEMM using Rayon.
///
/// This simulates multiple thread blocks running in parallel.
pub fn parallel_tiled_gemm(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    tile_size: usize,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // Generate all tile coordinates
    let num_tiles_m = (m + tile_size - 1) / tile_size;
    let num_tiles_n = (n + tile_size - 1) / tile_size;

    // Process tiles in parallel
    let results: Vec<(usize, usize, Vec<f32>)> = (0..num_tiles_m * num_tiles_n)
        .into_par_iter()
        .map(|tile_idx| {
            let tile_row_idx = tile_idx / num_tiles_n;
            let tile_col_idx = tile_idx % num_tiles_n;
            let tile_row = tile_row_idx * tile_size;
            let tile_col = tile_col_idx * tile_size;

            let mut c_acc = vec![0.0f32; tile_size * tile_size];

            // Process this tile
            for tile_k in (0..k).step_by(tile_size) {
                for i in 0..tile_size {
                    for j in 0..tile_size {
                        let row = tile_row + i;
                        let col = tile_col + j;
                        if row >= m || col >= n {
                            continue;
                        }

                        let mut sum = 0.0f32;
                        for kk in 0..tile_size {
                            let k_idx = tile_k + kk;
                            if k_idx >= k {
                                break;
                            }
                            sum += a.get(row, k_idx) * b.get(k_idx, col);
                        }
                        c_acc[i * tile_size + j] += sum;
                    }
                }
            }

            (tile_row, tile_col, c_acc)
        })
        .collect();

    // Write results back to C
    for (tile_row, tile_col, c_acc) in results {
        for i in 0..tile_size {
            for j in 0..tile_size {
                let row = tile_row + i;
                let col = tile_col + j;
                if row < m && col < n {
                    c.set(row, col, c_acc[i * tile_size + j]);
                }
            }
        }
    }

    Ok(())
}

/// Double-buffered GEMM simulation.
///
/// This demonstrates software pipelining where loading and computation overlap.
pub fn double_buffered_gemm(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    tile_size: usize,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // Double buffer arrays
    let mut a_buffer = [
        vec![0.0f32; tile_size * tile_size],
        vec![0.0f32; tile_size * tile_size],
    ];
    let mut b_buffer = [
        vec![0.0f32; tile_size * tile_size],
        vec![0.0f32; tile_size * tile_size],
    ];

    for tile_row in (0..m).step_by(tile_size) {
        for tile_col in (0..n).step_by(tile_size) {
            let mut c_acc = vec![0.0f32; tile_size * tile_size];
            let mut buffer_idx = 0;

            // Prefetch first tiles
            load_tile(a, &mut a_buffer[0], tile_row, 0, tile_size, m, k);
            load_tile_b(b, &mut b_buffer[0], 0, tile_col, tile_size, k, n);

            let num_k_tiles = (k + tile_size - 1) / tile_size;

            for kt in 0..num_k_tiles {
                let tile_k = kt * tile_size;

                // Prefetch next tile while computing current (double buffering)
                if kt + 1 < num_k_tiles {
                    let next_idx = 1 - buffer_idx;
                    let next_k = (kt + 1) * tile_size;
                    load_tile(a, &mut a_buffer[next_idx], tile_row, next_k, tile_size, m, k);
                    load_tile_b(b, &mut b_buffer[next_idx], next_k, tile_col, tile_size, k, n);
                }

                // Compute on current buffer
                for i in 0..tile_size {
                    for j in 0..tile_size {
                        let mut sum = 0.0f32;
                        for kk in 0..tile_size {
                            sum += a_buffer[buffer_idx][i * tile_size + kk]
                                * b_buffer[buffer_idx][kk * tile_size + j];
                        }
                        c_acc[i * tile_size + j] += sum;
                    }
                }

                // Swap buffers
                buffer_idx = 1 - buffer_idx;
            }

            // Write results
            for i in 0..tile_size {
                for j in 0..tile_size {
                    let row = tile_row + i;
                    let col = tile_col + j;
                    if row < m && col < n {
                        c.set(row, col, c_acc[i * tile_size + j]);
                    }
                }
            }
        }
    }

    Ok(())
}

// Helper functions
fn load_tile(
    mat: &Matrix,
    buffer: &mut [f32],
    row_start: usize,
    col_start: usize,
    tile_size: usize,
    max_rows: usize,
    max_cols: usize,
) {
    for i in 0..tile_size {
        for j in 0..tile_size {
            let row = row_start + i;
            let col = col_start + j;
            buffer[i * tile_size + j] = if row < max_rows && col < max_cols {
                mat.get(row, col)
            } else {
                0.0
            };
        }
    }
}

fn load_tile_b(
    mat: &Matrix,
    buffer: &mut [f32],
    row_start: usize,
    col_start: usize,
    tile_size: usize,
    max_rows: usize,
    max_cols: usize,
) {
    load_tile(mat, buffer, row_start, col_start, tile_size, max_rows, max_cols);
}

/// GEMM with alpha/beta scaling: C = alpha * A * B + beta * C
pub fn scaled_gemm(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a.get(row, i) * b.get(i, col);
            }
            let old = c.get(row, col);
            c.set(row, col, alpha * sum + beta * old);
        }
    }

    Ok(())
}

// ============================================================================
// Batched GEMM Operations
// ============================================================================

/// Batched GEMM: Compute multiple independent matrix multiplications in parallel.
///
/// C[i] = A[i] * B[i] for i in 0..batch_size
///
/// This is useful for:
/// - Attention mechanisms in transformers
/// - Batch processing in neural networks
/// - Multi-head attention
pub fn batched_gemm(
    a_batch: &[Matrix],
    b_batch: &[Matrix],
    c_batch: &mut [Matrix],
) -> Result<()> {
    if a_batch.len() != b_batch.len() || a_batch.len() != c_batch.len() {
        return Err(Error::DimensionMismatch(
            "Batch sizes must match".into()
        ));
    }

    if a_batch.is_empty() {
        return Ok(());
    }

    // Validate dimensions for all matrices
    for i in 0..a_batch.len() {
        if a_batch[i].cols != b_batch[i].rows {
            return Err(Error::DimensionMismatch(format!(
                "Batch {}: A cols ({}) != B rows ({})",
                i, a_batch[i].cols, b_batch[i].rows
            )));
        }
        if c_batch[i].rows != a_batch[i].rows || c_batch[i].cols != b_batch[i].cols {
            return Err(Error::DimensionMismatch(format!(
                "Batch {}: C dimensions don't match result",
                i
            )));
        }
    }

    // Process batches in parallel
    c_batch
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, c)| {
            let a = &a_batch[i];
            let b = &b_batch[i];
            let m = a.rows;
            let n = b.cols;
            let k = a.cols;

            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a.get(row, kk) * b.get(kk, col);
                    }
                    c.set(row, col, sum);
                }
            }
        });

    Ok(())
}

/// Batched GEMM with scaling: C[i] = alpha * A[i] * B[i] + beta * C[i]
pub fn batched_scaled_gemm(
    a_batch: &[Matrix],
    b_batch: &[Matrix],
    c_batch: &mut [Matrix],
    alpha: f32,
    beta: f32,
) -> Result<()> {
    if a_batch.len() != b_batch.len() || a_batch.len() != c_batch.len() {
        return Err(Error::DimensionMismatch(
            "Batch sizes must match".into()
        ));
    }

    if a_batch.is_empty() {
        return Ok(());
    }

    c_batch
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, c)| {
            let a = &a_batch[i];
            let b = &b_batch[i];
            let m = a.rows;
            let n = b.cols;
            let k = a.cols;

            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a.get(row, kk) * b.get(kk, col);
                    }
                    let old = c.get(row, col);
                    c.set(row, col, alpha * sum + beta * old);
                }
            }
        });

    Ok(())
}

/// Strided batched GEMM for contiguous memory layouts.
///
/// Operates on matrices stored in a single contiguous buffer with strides.
/// This is common in deep learning frameworks where batched matrices are
/// stored as 3D tensors.
pub fn strided_batched_gemm(
    a_data: &[f32],
    b_data: &[f32],
    c_data: &mut [f32],
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize, // Elements between A matrices
    stride_b: usize, // Elements between B matrices
    stride_c: usize, // Elements between C matrices
) -> Result<()> {
    if a_data.len() < batch_size * stride_a {
        return Err(Error::DimensionMismatch("A data too small for batch".into()));
    }
    if b_data.len() < batch_size * stride_b {
        return Err(Error::DimensionMismatch("B data too small for batch".into()));
    }
    if c_data.len() < batch_size * stride_c {
        return Err(Error::DimensionMismatch("C data too small for batch".into()));
    }

    // Process batches in parallel
    (0..batch_size).into_par_iter().for_each(|batch| {
        let a_offset = batch * stride_a;
        let b_offset = batch * stride_b;
        let c_offset = batch * stride_c;

        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_idx = a_offset + row * k + kk;
                    let b_idx = b_offset + kk * n + col;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                let c_idx = c_offset + row * n + col;
                // Note: We need unsafe here for parallel write, but indices don't overlap
                unsafe {
                    let c_ptr = c_data.as_ptr() as *mut f32;
                    *c_ptr.add(c_idx) = sum;
                }
            }
        }
    });

    Ok(())
}

// ============================================================================
// Fused GEMM Operations
// ============================================================================

/// Activation function types for fused operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// No activation (identity)
    None,
    /// ReLU: max(0, x)
    ReLU,
    /// Leaky ReLU: x if x > 0 else alpha * x
    LeakyReLU(f32),
    /// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    GeLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh
    Tanh,
    /// SiLU (Swish): x * sigmoid(x)
    SiLU,
}

impl Activation {
    /// Apply activation function to a value.
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            Activation::GeLU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_over_pi = 0.7978845608028654f32;
                let coeff = 0.044715f32;
                let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::SiLU => x * (1.0 / (1.0 + (-x).exp())),
        }
    }
}

/// Fused GEMM + Activation: C = activation(A * B)
///
/// Fusing operations reduces memory bandwidth by avoiding intermediate writes.
pub fn gemm_activation(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    activation: Activation,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a.get(row, i) * b.get(i, col);
            }
            c.set(row, col, activation.apply(sum));
        }
    }

    Ok(())
}

/// Fused GEMM + Bias + Activation: C = activation(A * B + bias)
///
/// Common in neural network layers (Linear + ReLU, etc.)
pub fn gemm_bias_activation(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    bias: &[f32],
    activation: Activation,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }
    if bias.len() != b.cols {
        return Err(Error::DimensionMismatch(format!(
            "Bias length ({}) != output columns ({})",
            bias.len(), b.cols
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a.get(row, i) * b.get(i, col);
            }
            sum += bias[col];
            c.set(row, col, activation.apply(sum));
        }
    }

    Ok(())
}

/// Fused GEMM + scaled output: C = alpha * activation(A * B + bias) + beta * C
///
/// Full BLAS-like interface with activation fusion.
pub fn gemm_fused(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    alpha: f32,
    beta: f32,
    bias: Option<&[f32]>,
    activation: Activation,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }
    if let Some(b_vec) = bias {
        if b_vec.len() != b.cols {
            return Err(Error::DimensionMismatch(format!(
                "Bias length ({}) != output columns ({})",
                b_vec.len(), b.cols
            )));
        }
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a.get(row, i) * b.get(i, col);
            }
            if let Some(b_vec) = bias {
                sum += b_vec[col];
            }
            let activated = activation.apply(sum);
            let old = c.get(row, col);
            c.set(row, col, alpha * activated + beta * old);
        }
    }

    Ok(())
}

// ============================================================================
// Tensor Core / WMMA Simulation
// ============================================================================

/// Simulated Tensor Core fragment sizes (matching NVIDIA Tensor Cores).
#[derive(Debug, Clone, Copy)]
pub struct WmmaConfig {
    /// M dimension of the fragment
    pub m: usize,
    /// N dimension of the fragment
    pub n: usize,
    /// K dimension of the fragment
    pub k: usize,
}

impl Default for WmmaConfig {
    fn default() -> Self {
        // Standard 16x16x16 configuration
        Self { m: 16, n: 16, k: 16 }
    }
}

impl WmmaConfig {
    /// 16x16x16 configuration (FP16)
    pub fn m16n16k16() -> Self {
        Self { m: 16, n: 16, k: 16 }
    }

    /// 8x32x16 configuration
    pub fn m8n32k16() -> Self {
        Self { m: 8, n: 32, k: 16 }
    }

    /// 32x8x16 configuration
    pub fn m32n8k16() -> Self {
        Self { m: 32, n: 8, k: 16 }
    }
}

/// WMMA Fragment - simulates tensor core matrix fragments.
///
/// In actual GPU code, fragments are stored in warp-distributed registers.
/// Here we simulate the concept for educational purposes.
#[derive(Clone)]
pub struct WmmaFragment {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl WmmaFragment {
    /// Create a new fragment with given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Load fragment from matrix.
    pub fn load_matrix_sync(&mut self, matrix: &Matrix, row_offset: usize, col_offset: usize) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let row = row_offset + i;
                let col = col_offset + j;
                let val = if row < matrix.rows && col < matrix.cols {
                    matrix.get(row, col)
                } else {
                    0.0
                };
                self.data[i * self.cols + j] = val;
            }
        }
    }

    /// Store fragment to matrix.
    pub fn store_matrix_sync(&self, matrix: &mut Matrix, row_offset: usize, col_offset: usize) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let row = row_offset + i;
                let col = col_offset + j;
                if row < matrix.rows && col < matrix.cols {
                    matrix.set(row, col, self.data[i * self.cols + j]);
                }
            }
        }
    }

    /// Fill fragment with a constant value.
    pub fn fill(&mut self, value: f32) {
        for x in &mut self.data {
            *x = value;
        }
    }

    /// Get element at position.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at position.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }
}

/// Simulated WMMA matrix multiply-accumulate operation.
///
/// Computes D = A * B + C where fragments have WMMA dimensions.
/// This simulates the `wmma::mma_sync` CUDA operation.
pub fn wmma_mma_sync(
    a_frag: &WmmaFragment,
    b_frag: &WmmaFragment,
    c_frag: &WmmaFragment,
    d_frag: &mut WmmaFragment,
) -> Result<()> {
    let m = a_frag.rows;
    let n = b_frag.cols;
    let k = a_frag.cols;

    if b_frag.rows != k || c_frag.rows != m || c_frag.cols != n {
        return Err(Error::DimensionMismatch(
            "WMMA fragment dimensions incompatible".into()
        ));
    }

    // D = A * B + C
    for i in 0..m {
        for j in 0..n {
            let mut sum = c_frag.get(i, j);
            for kk in 0..k {
                sum += a_frag.get(i, kk) * b_frag.get(kk, j);
            }
            d_frag.set(i, j, sum);
        }
    }

    Ok(())
}

/// WMMA GEMM: Matrix multiplication using simulated Tensor Cores.
///
/// This demonstrates how modern GPUs use Tensor Cores for matrix operations.
/// The computation is tiled into WMMA fragments and processed using
/// matrix multiply-accumulate operations.
pub fn wmma_gemm(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    config: WmmaConfig,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch(format!(
            "A cols ({}) != B rows ({})",
            a.cols, b.rows
        )));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    let wmma_m = config.m;
    let wmma_n = config.n;
    let wmma_k = config.k;

    // Process tiles
    for tile_m in (0..m).step_by(wmma_m) {
        for tile_n in (0..n).step_by(wmma_n) {
            // Accumulator fragment
            let mut c_frag = WmmaFragment::new(wmma_m, wmma_n);
            c_frag.fill(0.0);

            // Load initial C values
            c_frag.load_matrix_sync(c, tile_m, tile_n);

            // Accumulate over K dimension
            for tile_k in (0..k).step_by(wmma_k) {
                // Load A and B fragments
                let mut a_frag = WmmaFragment::new(wmma_m, wmma_k);
                let mut b_frag = WmmaFragment::new(wmma_k, wmma_n);

                a_frag.load_matrix_sync(a, tile_m, tile_k);
                b_frag.load_matrix_sync(b, tile_k, tile_n);

                // WMMA MMA operation
                let mut d_frag = WmmaFragment::new(wmma_m, wmma_n);
                wmma_mma_sync(&a_frag, &b_frag, &c_frag, &mut d_frag)?;
                c_frag = d_frag;
            }

            // Store result
            c_frag.store_matrix_sync(c, tile_m, tile_n);
        }
    }

    Ok(())
}

// ============================================================================
// Roofline Analysis
// ============================================================================

/// Performance bound classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceBound {
    /// Limited by compute throughput (FLOPs)
    ComputeBound,
    /// Limited by memory bandwidth
    MemoryBound,
    /// Near the ridge point (balanced)
    Balanced,
}

/// Roofline model for performance analysis.
#[derive(Debug, Clone)]
pub struct RooflineModel {
    /// Peak compute in GFLOPs
    pub peak_gflops: f64,
    /// Peak memory bandwidth in GB/s
    pub peak_bandwidth_gbs: f64,
}

impl Default for RooflineModel {
    fn default() -> Self {
        // Typical GPU specs (adjust for actual hardware)
        Self {
            peak_gflops: 10000.0,      // 10 TFLOPs
            peak_bandwidth_gbs: 900.0,  // 900 GB/s
        }
    }
}

impl RooflineModel {
    /// Create a new roofline model with specified hardware characteristics.
    pub fn new(peak_gflops: f64, peak_bandwidth_gbs: f64) -> Self {
        Self { peak_gflops, peak_bandwidth_gbs }
    }

    /// Calculate the ridge point (arithmetic intensity where compute and memory balance).
    pub fn ridge_point(&self) -> f64 {
        self.peak_gflops / self.peak_bandwidth_gbs
    }

    /// Calculate theoretical peak performance for given arithmetic intensity.
    pub fn theoretical_peak(&self, arithmetic_intensity: f64) -> f64 {
        let memory_bound_perf = arithmetic_intensity * self.peak_bandwidth_gbs;
        memory_bound_perf.min(self.peak_gflops)
    }

    /// Determine performance bound for a given arithmetic intensity.
    pub fn classify_bound(&self, arithmetic_intensity: f64) -> PerformanceBound {
        let ridge = self.ridge_point();
        let ratio = arithmetic_intensity / ridge;

        if ratio < 0.5 {
            PerformanceBound::MemoryBound
        } else if ratio > 2.0 {
            PerformanceBound::ComputeBound
        } else {
            PerformanceBound::Balanced
        }
    }

    /// Calculate arithmetic intensity for GEMM: FLOPs / Bytes accessed.
    ///
    /// For C = A * B where A is MxK and B is KxN:
    /// - FLOPs = 2 * M * N * K (multiply + add per output element)
    /// - Bytes = 4 * (M*K + K*N + M*N) for FP32
    pub fn gemm_arithmetic_intensity(m: usize, n: usize, k: usize) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let bytes = 4.0 * ((m * k + k * n + m * n) as f64);
        flops / bytes
    }

    /// Calculate arithmetic intensity with tiling.
    ///
    /// Tiling improves arithmetic intensity by reusing data in cache.
    pub fn tiled_gemm_arithmetic_intensity(
        m: usize,
        n: usize,
        k: usize,
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
    ) -> f64 {
        // FLOPs stay the same
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);

        // With tiling, each tile of A is reused across N/tile_n tiles
        // and each tile of B is reused across M/tile_m tiles
        let num_m_tiles = (m + tile_m - 1) / tile_m;
        let num_n_tiles = (n + tile_n - 1) / tile_n;
        let num_k_tiles = (k + tile_k - 1) / tile_k;

        // Bytes loaded (assuming perfect caching within tiles):
        // A tiles: num_m_tiles * num_k_tiles * tile_m * tile_k * 4
        // B tiles: num_k_tiles * num_n_tiles * tile_k * tile_n * 4
        // C tiles: num_m_tiles * num_n_tiles * tile_m * tile_n * 4 (write)
        let a_bytes = (num_m_tiles * num_k_tiles * tile_m * tile_k) as f64 * 4.0;
        let b_bytes = (num_k_tiles * num_n_tiles * tile_k * tile_n) as f64 * 4.0;
        let c_bytes = (num_m_tiles * num_n_tiles * tile_m * tile_n) as f64 * 4.0;

        let total_bytes = a_bytes + b_bytes + c_bytes;
        flops / total_bytes
    }

    /// Analyze GEMM performance and return analysis results.
    pub fn analyze_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        achieved_gflops: f64,
    ) -> RooflineAnalysis {
        let arithmetic_intensity = Self::gemm_arithmetic_intensity(m, n, k);
        let theoretical_peak = self.theoretical_peak(arithmetic_intensity);
        let bound = self.classify_bound(arithmetic_intensity);
        let efficiency = achieved_gflops / theoretical_peak * 100.0;

        RooflineAnalysis {
            arithmetic_intensity,
            theoretical_peak_gflops: theoretical_peak,
            achieved_gflops,
            efficiency_percent: efficiency,
            bound,
            ridge_point: self.ridge_point(),
        }
    }
}

/// Results of roofline analysis.
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    /// Calculated arithmetic intensity (FLOPs/byte)
    pub arithmetic_intensity: f64,
    /// Theoretical peak performance for this workload
    pub theoretical_peak_gflops: f64,
    /// Actually achieved performance
    pub achieved_gflops: f64,
    /// Efficiency as percentage of theoretical peak
    pub efficiency_percent: f64,
    /// Performance bound classification
    pub bound: PerformanceBound,
    /// Ridge point of the hardware
    pub ridge_point: f64,
}

impl RooflineAnalysis {
    /// Get optimization recommendations based on analysis.
    pub fn recommendations(&self) -> Vec<&'static str> {
        let mut recs = Vec::new();

        match self.bound {
            PerformanceBound::MemoryBound => {
                recs.push("Increase tiling to improve data reuse");
                recs.push("Consider using shared memory for intermediate results");
                recs.push("Ensure coalesced memory access patterns");
            }
            PerformanceBound::ComputeBound => {
                recs.push("Optimize inner loops for instruction throughput");
                recs.push("Consider using Tensor Cores for mixed-precision");
                recs.push("Maximize occupancy for latency hiding");
            }
            PerformanceBound::Balanced => {
                recs.push("Near optimal; focus on fine-tuning");
                recs.push("Profile for specific bottlenecks");
            }
        }

        if self.efficiency_percent < 50.0 {
            recs.push("Low efficiency - check for bank conflicts");
            recs.push("Verify proper memory alignment");
        }

        recs
    }
}
