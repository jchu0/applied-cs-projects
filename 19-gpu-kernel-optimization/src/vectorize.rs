//! Vectorized memory operations for GPU kernel optimization.
//!
//! This module provides vectorized load/store operations that simulate
//! GPU vector memory access patterns (e.g., LDG.128, STG.128) for
//! optimal memory bandwidth utilization.

// Note: std::simd is unstable. Using manual implementations instead.

/// Vectorized load result for 4 floats.
#[derive(Debug, Clone, Copy)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Float4 {
    /// Create new Float4 from array.
    pub fn new(data: [f32; 4]) -> Self {
        Self {
            x: data[0],
            y: data[1],
            z: data[2],
            w: data[3],
        }
    }

    /// Create zeroed Float4.
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
    }

    /// Convert to array.
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Add two Float4 values.
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }

    /// Multiply by scalar.
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }

    /// Fused multiply-add: self * a + b.
    pub fn fma(self, a: f32, b: Self) -> Self {
        Self {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
            z: self.z.mul_add(a, b.z),
            w: self.w.mul_add(a, b.w),
        }
    }
}

/// Vectorized load result for 8 floats.
#[derive(Debug, Clone, Copy)]
pub struct Float8 {
    pub data: [f32; 8],
}

impl Float8 {
    /// Create new Float8 from array.
    pub fn new(data: [f32; 8]) -> Self {
        Self { data }
    }

    /// Create zeroed Float8.
    pub fn zero() -> Self {
        Self { data: [0.0; 8] }
    }

    /// Add two Float8 values.
    pub fn add(self, other: Self) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.data[i] + other.data[i];
        }
        Self { data: result }
    }

    /// Multiply by scalar.
    pub fn scale(self, s: f32) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.data[i] * s;
        }
        Self { data: result }
    }
}

/// Vectorized memory operations trait.
pub trait VectorizedOps {
    /// Aligned load of 4 floats (16 bytes).
    /// Simulates LDG.128 instruction.
    fn load_float4(data: &[f32], offset: usize) -> Float4;

    /// Aligned load of 8 floats (32 bytes).
    /// Simulates AVX/AVX2 vector loads.
    fn load_float8(data: &[f32], offset: usize) -> Float8;

    /// Aligned store of 4 floats (16 bytes).
    /// Simulates STG.128 instruction.
    fn store_float4(data: &mut [f32], offset: usize, value: Float4);

    /// Aligned store of 8 floats (32 bytes).
    fn store_float8(data: &mut [f32], offset: usize, value: Float8);

    /// Check if address is aligned for vector access.
    fn is_aligned(offset: usize, alignment: usize) -> bool;
}

/// SIMD-accelerated vector operations.
pub struct SimdVectorOps;

impl VectorizedOps for SimdVectorOps {
    fn load_float4(data: &[f32], offset: usize) -> Float4 {
        assert!(offset + 4 <= data.len(), "Out of bounds load_float4");
        Float4::new([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }

    fn load_float8(data: &[f32], offset: usize) -> Float8 {
        assert!(offset + 8 <= data.len(), "Out of bounds load_float8");
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(&data[offset..offset + 8]);
        Float8::new(arr)
    }

    fn store_float4(data: &mut [f32], offset: usize, value: Float4) {
        assert!(offset + 4 <= data.len(), "Out of bounds store_float4");
        data[offset] = value.x;
        data[offset + 1] = value.y;
        data[offset + 2] = value.z;
        data[offset + 3] = value.w;
    }

    fn store_float8(data: &mut [f32], offset: usize, value: Float8) {
        assert!(offset + 8 <= data.len(), "Out of bounds store_float8");
        data[offset..offset + 8].copy_from_slice(&value.data);
    }

    fn is_aligned(offset: usize, alignment: usize) -> bool {
        offset % alignment == 0
    }
}

/// Vectorized GEMM kernel using Float4 loads.
pub fn vectorized_gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Ensure k is divisible by 4 for vectorized access
    let k_vec = k / 4 * 4;
    let k_tail = k - k_vec;

    for i in 0..m {
        for j in 0..n {
            let mut sum = Float4::zero();
            let mut sum_scalar = 0.0f32;

            // Vectorized inner loop
            for kk in (0..k_vec).step_by(4) {
                let a_vec = SimdVectorOps::load_float4(a, i * k + kk);
                let b_vec = Float4::new([
                    b[kk * n + j],
                    b[(kk + 1) * n + j],
                    b[(kk + 2) * n + j],
                    b[(kk + 3) * n + j],
                ]);

                // Element-wise multiply-accumulate
                sum = Float4 {
                    x: sum.x + a_vec.x * b_vec.x,
                    y: sum.y + a_vec.y * b_vec.y,
                    z: sum.z + a_vec.z * b_vec.z,
                    w: sum.w + a_vec.w * b_vec.w,
                };
            }

            // Handle tail elements
            for kk in k_vec..k {
                sum_scalar += a[i * k + kk] * b[kk * n + j];
            }

            // Reduce Float4 to scalar
            c[i * n + j] = sum.x + sum.y + sum.z + sum.w + sum_scalar;
        }
    }
}

/// Vectorized GEMM with transposed B matrix (more coalesced access).
pub fn vectorized_gemm_transposed_b(
    a: &[f32],
    b_t: &[f32],  // B is already transposed
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let k_vec = k / 4 * 4;

    for i in 0..m {
        for j in 0..n {
            let mut sum = Float4::zero();
            let mut sum_scalar = 0.0f32;

            // Both A[i,:] and B_T[j,:] are contiguous, enabling coalesced loads
            for kk in (0..k_vec).step_by(4) {
                let a_vec = SimdVectorOps::load_float4(a, i * k + kk);
                let b_vec = SimdVectorOps::load_float4(b_t, j * k + kk);

                // SIMD multiply-accumulate
                sum = Float4 {
                    x: sum.x + a_vec.x * b_vec.x,
                    y: sum.y + a_vec.y * b_vec.y,
                    z: sum.z + a_vec.z * b_vec.z,
                    w: sum.w + a_vec.w * b_vec.w,
                };
            }

            // Handle tail
            for kk in k_vec..k {
                sum_scalar += a[i * k + kk] * b_t[j * k + kk];
            }

            c[i * n + j] = sum.x + sum.y + sum.z + sum.w + sum_scalar;
        }
    }
}

/// Memory coalescing analysis.
#[derive(Debug, Clone)]
pub struct CoalescingAnalysis {
    /// Number of fully coalesced accesses.
    pub coalesced_count: usize,
    /// Number of non-coalesced (strided) accesses.
    pub strided_count: usize,
    /// Average stride in elements.
    pub avg_stride: f32,
    /// Coalescing efficiency (0.0 to 1.0).
    pub efficiency: f32,
}

impl CoalescingAnalysis {
    /// Analyze memory access pattern for coalescing.
    pub fn analyze(
        access_pattern: &[(usize, usize)],  // (row, col) pairs
        row_major: bool,
        cols: usize,
    ) -> Self {
        if access_pattern.is_empty() {
            return Self {
                coalesced_count: 0,
                strided_count: 0,
                avg_stride: 0.0,
                efficiency: 1.0,
            };
        }

        let mut coalesced = 0;
        let mut strided = 0;
        let mut total_stride = 0usize;

        let to_linear = |r: usize, c: usize| -> usize {
            if row_major { r * cols + c } else { c * access_pattern.len() + r }
        };

        for i in 1..access_pattern.len() {
            let prev = to_linear(access_pattern[i - 1].0, access_pattern[i - 1].1);
            let curr = to_linear(access_pattern[i].0, access_pattern[i].1);

            let stride = if curr > prev { curr - prev } else { prev - curr };

            if stride <= 1 {
                coalesced += 1;
            } else {
                strided += 1;
                total_stride += stride;
            }
        }

        let total = coalesced + strided;
        let efficiency = if total > 0 {
            coalesced as f32 / total as f32
        } else {
            1.0
        };

        let avg_stride = if strided > 0 {
            total_stride as f32 / strided as f32
        } else {
            0.0
        };

        Self {
            coalesced_count: coalesced,
            strided_count: strided,
            avg_stride,
            efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float4_operations() {
        let a = Float4::new([1.0, 2.0, 3.0, 4.0]);
        let b = Float4::new([5.0, 6.0, 7.0, 8.0]);

        let sum = a.add(b);
        assert_eq!(sum.to_array(), [6.0, 8.0, 10.0, 12.0]);

        let scaled = a.scale(2.0);
        assert_eq!(scaled.to_array(), [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_load_store_float4() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let v = SimdVectorOps::load_float4(&data, 0);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);

        let v = SimdVectorOps::load_float4(&data, 4);
        assert_eq!(v.to_array(), [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_vectorized_gemm() {
        // 2x2 * 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        // Test the vectorized_gemm function directly
        vectorized_gemm(&a, &b, &mut c, 2, 2, 2);

        // Expected:
        // [1*5+2*7, 1*6+2*8]   = [19, 22]
        // [3*5+4*7, 3*6+4*8]   = [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_coalescing_analysis() {
        // Perfect coalescing: consecutive accesses
        let pattern: Vec<(usize, usize)> = (0..8).map(|i| (0, i)).collect();
        let analysis = CoalescingAnalysis::analyze(&pattern, true, 8);

        assert!(analysis.efficiency > 0.9);
        assert_eq!(analysis.strided_count, 0);
    }

    #[test]
    fn test_strided_access() {
        // Column access in row-major (stride = cols)
        let pattern: Vec<(usize, usize)> = (0..4).map(|i| (i, 0)).collect();
        let analysis = CoalescingAnalysis::analyze(&pattern, true, 8);

        assert!(analysis.efficiency < 0.5);
        assert!(analysis.avg_stride > 1.0);
    }

    #[test]
    fn test_float8_operations() {
        let a = Float8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Float8::new([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let sum = a.add(b);
        assert_eq!(sum.data, [9.0; 8]);
    }

    #[test]
    fn test_alignment_check() {
        assert!(SimdVectorOps::is_aligned(0, 4));
        assert!(SimdVectorOps::is_aligned(16, 4));
        assert!(!SimdVectorOps::is_aligned(5, 4));
    }
}
