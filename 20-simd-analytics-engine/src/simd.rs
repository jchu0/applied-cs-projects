//! SIMD-style vectorized operations.
//!
//! These operations are designed to be auto-vectorized by the compiler
//! and process data in chunks for cache efficiency.

use crate::{Result, BLOCK_SIZE, VECTOR_WIDTH};

/// SIMD operations for vectorized computation.
pub struct SimdOps;

impl SimdOps {
    /// Vectorized sum of f32 array.
    #[inline]
    pub fn sum_f32(data: &[f32]) -> f32 {
        // Process in chunks that can be auto-vectorized
        let chunks = data.chunks_exact(VECTOR_WIDTH);
        let remainder = chunks.remainder();

        // Accumulate in vector-width accumulators
        let mut acc = [0.0f32; VECTOR_WIDTH];

        for chunk in chunks {
            for i in 0..VECTOR_WIDTH {
                acc[i] += chunk[i];
            }
        }

        // Horizontal sum of accumulators
        let mut sum: f32 = acc.iter().sum();

        // Handle remainder
        for &val in remainder {
            sum += val;
        }

        sum
    }

    /// Vectorized sum of f64 array.
    #[inline]
    pub fn sum_f64(data: &[f64]) -> f64 {
        let vector_width = 4; // 256-bit / 64-bit
        let chunks = data.chunks_exact(vector_width);
        let remainder = chunks.remainder();

        let mut acc = [0.0f64; 4];

        for chunk in chunks {
            for i in 0..4 {
                acc[i] += chunk[i];
            }
        }

        let mut sum: f64 = acc.iter().sum();
        for &val in remainder {
            sum += val;
        }

        sum
    }

    /// Vectorized sum of i64 array.
    #[inline]
    pub fn sum_i64(data: &[i64]) -> i64 {
        let vector_width = 4;
        let chunks = data.chunks_exact(vector_width);
        let remainder = chunks.remainder();

        let mut acc = [0i64; 4];

        for chunk in chunks {
            for i in 0..4 {
                acc[i] += chunk[i];
            }
        }

        let mut sum: i64 = acc.iter().sum();
        for &val in remainder {
            sum += val;
        }

        sum
    }

    /// Vectorized min of f32 array.
    #[inline]
    pub fn min_f32(data: &[f32]) -> Option<f32> {
        if data.is_empty() {
            return None;
        }

        let chunks = data.chunks_exact(VECTOR_WIDTH);
        let remainder = chunks.remainder();

        let mut mins = [f32::INFINITY; VECTOR_WIDTH];

        for chunk in chunks {
            for i in 0..VECTOR_WIDTH {
                mins[i] = mins[i].min(chunk[i]);
            }
        }

        let mut result = mins.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        for &val in remainder {
            result = result.min(val);
        }

        Some(result)
    }

    /// Vectorized max of f32 array.
    #[inline]
    pub fn max_f32(data: &[f32]) -> Option<f32> {
        if data.is_empty() {
            return None;
        }

        let chunks = data.chunks_exact(VECTOR_WIDTH);
        let remainder = chunks.remainder();

        let mut maxs = [f32::NEG_INFINITY; VECTOR_WIDTH];

        for chunk in chunks {
            for i in 0..VECTOR_WIDTH {
                maxs[i] = maxs[i].max(chunk[i]);
            }
        }

        let mut result = maxs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for &val in remainder {
            result = result.max(val);
        }

        Some(result)
    }

    /// Vectorized min of f64 array.
    #[inline]
    pub fn min_f64(data: &[f64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }

        let vector_width = 4;
        let chunks = data.chunks_exact(vector_width);
        let remainder = chunks.remainder();

        let mut mins = [f64::INFINITY; 4];

        for chunk in chunks {
            for i in 0..4 {
                mins[i] = mins[i].min(chunk[i]);
            }
        }

        let mut result = mins.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        for &val in remainder {
            result = result.min(val);
        }

        Some(result)
    }

    /// Vectorized max of f64 array.
    #[inline]
    pub fn max_f64(data: &[f64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }

        let vector_width = 4;
        let chunks = data.chunks_exact(vector_width);
        let remainder = chunks.remainder();

        let mut maxs = [f64::NEG_INFINITY; 4];

        for chunk in chunks {
            for i in 0..4 {
                maxs[i] = maxs[i].max(chunk[i]);
            }
        }

        let mut result = maxs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        for &val in remainder {
            result = result.max(val);
        }

        Some(result)
    }

    /// Vectorized element-wise addition.
    #[inline]
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len().min(b.len()).min(out.len());

        for i in 0..len {
            out[i] = a[i] + b[i];
        }
    }

    /// Vectorized element-wise multiplication.
    #[inline]
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len().min(b.len()).min(out.len());

        for i in 0..len {
            out[i] = a[i] * b[i];
        }
    }

    /// Vectorized fused multiply-add: out = a * b + c.
    #[inline]
    pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len()).min(out.len());

        for i in 0..len {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    /// Vectorized scalar multiplication.
    #[inline]
    pub fn scale_f32(data: &[f32], scalar: f32, out: &mut [f32]) {
        let len = data.len().min(out.len());

        for i in 0..len {
            out[i] = data[i] * scalar;
        }
    }

    /// Vectorized scalar multiplication for f64.
    #[inline]
    pub fn scale_f64(data: &[f64], scalar: f64, out: &mut [f64]) {
        let len = data.len().min(out.len());

        for i in 0..len {
            out[i] = data[i] * scalar;
        }
    }

    /// Vectorized comparison: count elements > threshold.
    #[inline]
    pub fn count_gt_f32(data: &[f32], threshold: f32) -> usize {
        let chunks = data.chunks_exact(VECTOR_WIDTH);
        let remainder = chunks.remainder();

        let mut counts = [0usize; VECTOR_WIDTH];

        for chunk in chunks {
            for i in 0..VECTOR_WIDTH {
                if chunk[i] > threshold {
                    counts[i] += 1;
                }
            }
        }

        let mut count: usize = counts.iter().sum();
        for &val in remainder {
            if val > threshold {
                count += 1;
            }
        }

        count
    }

    /// Vectorized comparison: count elements < threshold.
    #[inline]
    pub fn count_lt_f32(data: &[f32], threshold: f32) -> usize {
        let chunks = data.chunks_exact(VECTOR_WIDTH);
        let remainder = chunks.remainder();

        let mut counts = [0usize; VECTOR_WIDTH];

        for chunk in chunks {
            for i in 0..VECTOR_WIDTH {
                if chunk[i] < threshold {
                    counts[i] += 1;
                }
            }
        }

        let mut count: usize = counts.iter().sum();
        for &val in remainder {
            if val < threshold {
                count += 1;
            }
        }

        count
    }

    /// Vectorized conditional sum (sum where mask is true).
    #[inline]
    pub fn masked_sum_f32(data: &[f32], mask: &[bool]) -> f32 {
        let len = data.len().min(mask.len());
        let mut sum = 0.0f32;

        for i in 0..len {
            if mask[i] {
                sum += data[i];
            }
        }

        sum
    }

    /// Vectorized dot product.
    #[inline]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let chunks_a = a[..len].chunks_exact(VECTOR_WIDTH);
        let chunks_b = b[..len].chunks_exact(VECTOR_WIDTH);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc = [0.0f32; VECTOR_WIDTH];

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            for i in 0..VECTOR_WIDTH {
                acc[i] += chunk_a[i] * chunk_b[i];
            }
        }

        let mut sum: f32 = acc.iter().sum();
        for (&va, &vb) in remainder_a.iter().zip(remainder_b) {
            sum += va * vb;
        }

        sum
    }

    /// Vectorized dot product for f64.
    #[inline]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let vector_width = 4;
        let len = a.len().min(b.len());
        let chunks_a = a[..len].chunks_exact(vector_width);
        let chunks_b = b[..len].chunks_exact(vector_width);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc = [0.0f64; 4];

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            for i in 0..4 {
                acc[i] += chunk_a[i] * chunk_b[i];
            }
        }

        let mut sum: f64 = acc.iter().sum();
        for (&va, &vb) in remainder_a.iter().zip(remainder_b) {
            sum += va * vb;
        }

        sum
    }

    /// Cache-blocked operation for large arrays.
    #[inline]
    pub fn blocked_sum_f32(data: &[f32]) -> f32 {
        let mut total = 0.0f32;

        // Process in cache-friendly blocks
        for block in data.chunks(BLOCK_SIZE) {
            total += Self::sum_f32(block);
        }

        total
    }

    /// Vectorized hash function for i64 keys.
    #[inline]
    pub fn hash_i64(keys: &[i64], hashes: &mut [u64]) {
        let len = keys.len().min(hashes.len());

        for i in 0..len {
            let mut h = keys[i] as u64;
            // MurmurHash3 finalizer
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            hashes[i] = h;
        }
    }

    /// Vectorized gather operation.
    #[inline]
    pub fn gather_f32(data: &[f32], indices: &[usize], out: &mut [f32]) {
        let len = indices.len().min(out.len());

        for i in 0..len {
            let idx = indices[i];
            out[i] = if idx < data.len() { data[idx] } else { 0.0 };
        }
    }

    /// Vectorized scatter operation.
    #[inline]
    pub fn scatter_f32(data: &[f32], indices: &[usize], out: &mut [f32]) {
        for (i, &idx) in indices.iter().enumerate() {
            if idx < out.len() && i < data.len() {
                out[idx] = data[i];
            }
        }
    }
}

/// Horizontal operations across SIMD lanes.
pub struct HorizontalOps;

impl HorizontalOps {
    /// Horizontal sum of 8 f32 values.
    #[inline]
    pub fn hsum_8xf32(values: [f32; 8]) -> f32 {
        // Tree reduction
        let v4 = [
            values[0] + values[4],
            values[1] + values[5],
            values[2] + values[6],
            values[3] + values[7],
        ];
        let v2 = [v4[0] + v4[2], v4[1] + v4[3]];
        v2[0] + v2[1]
    }

    /// Horizontal sum of 4 f64 values.
    #[inline]
    pub fn hsum_4xf64(values: [f64; 4]) -> f64 {
        let v2 = [values[0] + values[2], values[1] + values[3]];
        v2[0] + v2[1]
    }

    /// Horizontal min of 8 f32 values.
    #[inline]
    pub fn hmin_8xf32(values: [f32; 8]) -> f32 {
        let v4 = [
            values[0].min(values[4]),
            values[1].min(values[5]),
            values[2].min(values[6]),
            values[3].min(values[7]),
        ];
        let v2 = [v4[0].min(v4[2]), v4[1].min(v4[3])];
        v2[0].min(v2[1])
    }

    /// Horizontal max of 8 f32 values.
    #[inline]
    pub fn hmax_8xf32(values: [f32; 8]) -> f32 {
        let v4 = [
            values[0].max(values[4]),
            values[1].max(values[5]),
            values[2].max(values[6]),
            values[3].max(values[7]),
        ];
        let v2 = [v4[0].max(v4[2]), v4[1].max(v4[3])];
        v2[0].max(v2[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let sum = SimdOps::sum_f32(&data);
        let expected: f32 = (0..1000).map(|i| i as f32).sum();
        assert!((sum - expected).abs() < 0.001);
    }

    #[test]
    fn test_sum_f64() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let sum = SimdOps::sum_f64(&data);
        let expected: f64 = (0..1000).map(|i| i as f64).sum();
        assert!((sum - expected).abs() < 0.001);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_eq!(SimdOps::min_f32(&data), Some(1.0));
        assert_eq!(SimdOps::max_f32(&data), Some(9.0));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let dot = SimdOps::dot_f32(&a, &b);
        assert_eq!(dot, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_count_gt() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(SimdOps::count_gt_f32(&data, 5.0), 5);
    }

    #[test]
    fn test_horizontal_sum() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(HorizontalOps::hsum_8xf32(values), 36.0);
    }
}
