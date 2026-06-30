//! Advanced optimization techniques for analytics operations.
//!
//! This module provides cache-blocking, prefetching, and branch-free algorithms.

use crate::{Error, Result, BLOCK_SIZE, CACHE_LINE_SIZE, PREFETCH_DISTANCE, VECTOR_WIDTH};

/// Cache blocking configuration.
#[derive(Debug, Clone)]
pub struct CacheBlockConfig {
    /// L1 cache block size (in elements).
    pub l1_block_size: usize,
    /// L2 cache block size (in elements).
    pub l2_block_size: usize,
    /// L3 cache block size (in elements).
    pub l3_block_size: usize,
    /// Prefetch distance in cache lines.
    pub prefetch_distance: usize,
}

impl Default for CacheBlockConfig {
    fn default() -> Self {
        // Default sizes optimized for typical cache hierarchies
        Self {
            l1_block_size: 4096,      // ~32KB for f64
            l2_block_size: 32768,     // ~256KB for f64
            l3_block_size: 1048576,   // ~8MB for f64
            prefetch_distance: PREFETCH_DISTANCE,
        }
    }
}

impl CacheBlockConfig {
    /// Create config for specific cache sizes.
    pub fn for_cache_sizes(l1_kb: usize, l2_kb: usize, l3_kb: usize) -> Self {
        Self {
            l1_block_size: (l1_kb * 1024) / std::mem::size_of::<f64>() / 2, // Use half L1
            l2_block_size: (l2_kb * 1024) / std::mem::size_of::<f64>() / 2, // Use half L2
            l3_block_size: (l3_kb * 1024) / std::mem::size_of::<f64>() / 2, // Use half L3
            prefetch_distance: PREFETCH_DISTANCE,
        }
    }
}

/// Cache-blocked sum operation.
pub struct CacheBlockedOps;

impl CacheBlockedOps {
    /// Blocked sum of f64 values with prefetching hints.
    pub fn blocked_sum_f64(data: &[f64], config: &CacheBlockConfig) -> f64 {
        let mut total = 0.0f64;

        // Process in L1-sized blocks
        for block in data.chunks(config.l1_block_size) {
            let mut block_sum = 0.0f64;

            // SIMD-friendly loop
            for chunk in block.chunks(VECTOR_WIDTH) {
                for &val in chunk {
                    block_sum += val;
                }
            }

            total += block_sum;
        }

        total
    }

    /// Blocked sum with explicit prefetching (simulated).
    pub fn blocked_sum_f64_prefetch(data: &[f64], config: &CacheBlockConfig) -> f64 {
        let mut total = 0.0f64;
        let prefetch_elements = config.prefetch_distance * CACHE_LINE_SIZE / std::mem::size_of::<f64>();

        for (i, block) in data.chunks(config.l1_block_size).enumerate() {
            // Prefetch next block (simulated - actual prefetch would use intrinsics)
            let _prefetch_offset = (i + 1) * config.l1_block_size;

            let mut block_sum = 0.0f64;
            let mut acc = [0.0f64; 4];

            // Unrolled accumulation
            for chunk in block.chunks_exact(4) {
                acc[0] += chunk[0];
                acc[1] += chunk[1];
                acc[2] += chunk[2];
                acc[3] += chunk[3];
            }

            // Handle remainder
            let remainder = block.chunks_exact(4).remainder();
            for &val in remainder {
                block_sum += val;
            }

            block_sum += acc[0] + acc[1] + acc[2] + acc[3];
            total += block_sum;
        }

        total
    }

    /// Blocked min of f64 values.
    pub fn blocked_min_f64(data: &[f64], config: &CacheBlockConfig) -> Option<f64> {
        if data.is_empty() {
            return None;
        }

        let mut global_min = f64::INFINITY;

        for block in data.chunks(config.l1_block_size) {
            let mut mins = [f64::INFINITY; 4];

            for chunk in block.chunks_exact(4) {
                mins[0] = mins[0].min(chunk[0]);
                mins[1] = mins[1].min(chunk[1]);
                mins[2] = mins[2].min(chunk[2]);
                mins[3] = mins[3].min(chunk[3]);
            }

            let remainder = block.chunks_exact(4).remainder();
            let mut block_min = mins[0].min(mins[1]).min(mins[2]).min(mins[3]);
            for &val in remainder {
                block_min = block_min.min(val);
            }

            global_min = global_min.min(block_min);
        }

        Some(global_min)
    }

    /// Blocked max of f64 values.
    pub fn blocked_max_f64(data: &[f64], config: &CacheBlockConfig) -> Option<f64> {
        if data.is_empty() {
            return None;
        }

        let mut global_max = f64::NEG_INFINITY;

        for block in data.chunks(config.l1_block_size) {
            let mut maxs = [f64::NEG_INFINITY; 4];

            for chunk in block.chunks_exact(4) {
                maxs[0] = maxs[0].max(chunk[0]);
                maxs[1] = maxs[1].max(chunk[1]);
                maxs[2] = maxs[2].max(chunk[2]);
                maxs[3] = maxs[3].max(chunk[3]);
            }

            let remainder = block.chunks_exact(4).remainder();
            let mut block_max = maxs[0].max(maxs[1]).max(maxs[2]).max(maxs[3]);
            for &val in remainder {
                block_max = block_max.max(val);
            }

            global_max = global_max.max(block_max);
        }

        Some(global_max)
    }

    /// Blocked dot product.
    pub fn blocked_dot_f64(a: &[f64], b: &[f64], config: &CacheBlockConfig) -> f64 {
        let len = a.len().min(b.len());
        let mut total = 0.0f64;

        for (a_block, b_block) in a.chunks(config.l1_block_size)
            .zip(b.chunks(config.l1_block_size))
        {
            let mut acc = [0.0f64; 4];
            let block_len = a_block.len().min(b_block.len());

            for i in (0..block_len).step_by(4) {
                if i + 4 <= block_len {
                    acc[0] += a_block[i] * b_block[i];
                    acc[1] += a_block[i + 1] * b_block[i + 1];
                    acc[2] += a_block[i + 2] * b_block[i + 2];
                    acc[3] += a_block[i + 3] * b_block[i + 3];
                } else {
                    for j in i..block_len {
                        acc[0] += a_block[j] * b_block[j];
                    }
                }
            }

            total += acc[0] + acc[1] + acc[2] + acc[3];
        }

        total
    }

    /// Blocked element-wise operation.
    pub fn blocked_map_f64<F>(data: &[f64], config: &CacheBlockConfig, f: F) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        let mut result = Vec::with_capacity(data.len());

        for block in data.chunks(config.l1_block_size) {
            for &val in block {
                result.push(f(val));
            }
        }

        result
    }
}

/// Branch-free operations for predictable performance.
pub struct BranchFreeOps;

impl BranchFreeOps {
    /// Branch-free min of two values.
    #[inline]
    pub fn min_f64(a: f64, b: f64) -> f64 {
        // Use built-in which compiles to efficient branchless code
        a.min(b)
    }

    /// Branch-free max of two values.
    #[inline]
    pub fn max_f64(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    /// Branch-free clamp.
    #[inline]
    pub fn clamp_f64(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }

    /// Branch-free absolute value.
    #[inline]
    pub fn abs_f64(value: f64) -> f64 {
        value.abs()
    }

    /// Branch-free sign function (-1, 0, or 1).
    #[inline]
    pub fn sign_f64(value: f64) -> f64 {
        if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// Branch-free select (like ternary operator).
    #[inline]
    pub fn select_f64(condition: bool, a: f64, b: f64) -> f64 {
        if condition { a } else { b }
    }

    /// Branch-free conditional sum.
    pub fn conditional_sum_f64(data: &[f64], threshold: f64) -> f64 {
        let mut sum = 0.0f64;
        for &val in data {
            // Branchless: multiply by 0 or 1
            let mask = if val > threshold { 1.0 } else { 0.0 };
            sum += val * mask;
        }
        sum
    }

    /// Branch-free conditional count.
    pub fn conditional_count_f64(data: &[f64], threshold: f64) -> usize {
        let mut count = 0usize;
        for &val in data {
            count += (val > threshold) as usize;
        }
        count
    }
}

/// Streaming operations for memory-bound workloads.
pub struct StreamingOps;

impl StreamingOps {
    /// Streaming sum with temporal hints (simulated).
    pub fn streaming_sum_f64(data: &[f64]) -> f64 {
        // For large datasets that don't fit in cache,
        // use streaming loads to avoid cache pollution
        let mut sum = 0.0f64;
        let mut acc = [0.0f64; VECTOR_WIDTH];

        for chunk in data.chunks_exact(VECTOR_WIDTH) {
            for i in 0..VECTOR_WIDTH {
                acc[i] += chunk[i];
            }
        }

        sum = acc.iter().sum();
        for &val in data.chunks_exact(VECTOR_WIDTH).remainder() {
            sum += val;
        }

        sum
    }

    /// Streaming write with non-temporal stores (simulated).
    pub fn streaming_fill_f64(data: &mut [f64], value: f64) {
        // In real code, would use _mm256_stream_pd
        for chunk in data.chunks_mut(VECTOR_WIDTH) {
            for slot in chunk {
                *slot = value;
            }
        }
    }

    /// Streaming copy with non-temporal operations (simulated).
    pub fn streaming_copy_f64(src: &[f64], dst: &mut [f64]) {
        let len = src.len().min(dst.len());

        for i in 0..len {
            dst[i] = src[i];
        }
    }
}

/// Loop tiling for multi-dimensional operations.
pub struct TiledOps;

impl TiledOps {
    /// Tiled matrix-vector multiply (for 1D storage of 2D data).
    pub fn tiled_matvec_f64(
        matrix: &[f64],
        vector: &[f64],
        result: &mut [f64],
        rows: usize,
        cols: usize,
        tile_size: usize,
    ) {
        assert_eq!(matrix.len(), rows * cols);
        assert_eq!(vector.len(), cols);
        assert_eq!(result.len(), rows);

        // Initialize result
        result.fill(0.0);

        // Process tiles
        for row_tile in (0..rows).step_by(tile_size) {
            for col_tile in (0..cols).step_by(tile_size) {
                let row_end = (row_tile + tile_size).min(rows);
                let col_end = (col_tile + tile_size).min(cols);

                for i in row_tile..row_end {
                    let mut sum = 0.0f64;
                    for j in col_tile..col_end {
                        sum += matrix[i * cols + j] * vector[j];
                    }
                    result[i] += sum;
                }
            }
        }
    }

    /// Tiled reduction for 2D data.
    pub fn tiled_reduction_f64(
        data: &[f64],
        rows: usize,
        cols: usize,
        tile_size: usize,
    ) -> Vec<f64> {
        assert_eq!(data.len(), rows * cols);

        let mut row_sums = vec![0.0f64; rows];

        for row_tile in (0..rows).step_by(tile_size) {
            for col_tile in (0..cols).step_by(tile_size) {
                let row_end = (row_tile + tile_size).min(rows);
                let col_end = (col_tile + tile_size).min(cols);

                for i in row_tile..row_end {
                    for j in col_tile..col_end {
                        row_sums[i] += data[i * cols + j];
                    }
                }
            }
        }

        row_sums
    }
}

/// Software prefetching utilities.
pub struct Prefetcher;

impl Prefetcher {
    /// Calculate prefetch distance for given data type and access pattern.
    pub fn calculate_distance<T>(access_pattern: AccessPattern) -> usize {
        let element_size = std::mem::size_of::<T>();
        let cache_line_elements = CACHE_LINE_SIZE / element_size;

        match access_pattern {
            AccessPattern::Sequential => PREFETCH_DISTANCE * cache_line_elements,
            AccessPattern::Strided(stride) => PREFETCH_DISTANCE * stride,
            AccessPattern::Random => 0, // Prefetching doesn't help for random access
        }
    }

    /// Get optimal chunk size for prefetching.
    pub fn optimal_chunk_size<T>() -> usize {
        let element_size = std::mem::size_of::<T>();
        let cache_line_elements = CACHE_LINE_SIZE / element_size;
        cache_line_elements * PREFETCH_DISTANCE
    }
}

/// Access pattern for prefetching.
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Sequential access.
    Sequential,
    /// Strided access with given stride.
    Strided(usize),
    /// Random access.
    Random,
}

/// Memory bandwidth optimizer.
pub struct BandwidthOptimizer {
    /// Estimated memory bandwidth in GB/s.
    pub bandwidth_gb_s: f64,
}

impl Default for BandwidthOptimizer {
    fn default() -> Self {
        Self::new(50.0) // Conservative default
    }
}

impl BandwidthOptimizer {
    /// Create with estimated bandwidth.
    pub fn new(bandwidth_gb_s: f64) -> Self {
        Self { bandwidth_gb_s }
    }

    /// Estimate time to process data.
    pub fn estimate_time(&self, bytes: usize) -> f64 {
        (bytes as f64) / (self.bandwidth_gb_s * 1e9)
    }

    /// Check if operation is memory-bound.
    pub fn is_memory_bound(&self, bytes: usize, flops: usize) -> bool {
        // Operational intensity = flops / bytes
        // Memory-bound if intensity < machine balance
        let intensity = flops as f64 / bytes as f64;
        let machine_balance = 10.0; // Typical for modern CPUs
        intensity < machine_balance
    }

    /// Calculate effective bandwidth utilization.
    pub fn bandwidth_utilization(&self, bytes: usize, time_seconds: f64) -> f64 {
        let achieved_bandwidth = (bytes as f64) / (time_seconds * 1e9);
        (achieved_bandwidth / self.bandwidth_gb_s).min(1.0)
    }
}

/// Auto-tuner for selecting optimal algorithms.
#[derive(Debug)]
pub struct AutoTuner {
    /// Cache configuration.
    config: CacheBlockConfig,
    /// History of measurements.
    measurements: Vec<(String, usize, f64)>,
}

impl AutoTuner {
    /// Create new auto-tuner.
    pub fn new() -> Self {
        Self {
            config: CacheBlockConfig::default(),
            measurements: Vec::new(),
        }
    }

    /// Get optimal block size for data size.
    pub fn optimal_block_size(&self, data_size: usize) -> usize {
        if data_size <= self.config.l1_block_size {
            data_size
        } else if data_size <= self.config.l2_block_size {
            self.config.l1_block_size
        } else {
            self.config.l2_block_size
        }
    }

    /// Should use parallel execution?
    pub fn should_parallelize(&self, data_size: usize, operation_cost: usize) -> bool {
        // Parallelize if enough work
        let total_work = data_size * operation_cost;
        total_work > 100_000 // Threshold for parallelization overhead
    }

    /// Record a measurement.
    pub fn record(&mut self, operation: &str, data_size: usize, time_seconds: f64) {
        self.measurements.push((operation.to_string(), data_size, time_seconds));
    }

    /// Get average throughput for operation.
    pub fn average_throughput(&self, operation: &str) -> Option<f64> {
        let matching: Vec<_> = self.measurements
            .iter()
            .filter(|(op, _, _)| op == operation)
            .collect();

        if matching.is_empty() {
            return None;
        }

        let total_elements: usize = matching.iter().map(|(_, size, _)| size).sum();
        let total_time: f64 = matching.iter().map(|(_, _, time)| time).sum();

        Some(total_elements as f64 / total_time)
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_block_config() {
        let config = CacheBlockConfig::default();
        assert!(config.l1_block_size > 0);
        assert!(config.l2_block_size > config.l1_block_size);
        assert!(config.l3_block_size > config.l2_block_size);
    }

    #[test]
    fn test_blocked_sum() {
        let config = CacheBlockConfig::default();
        let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();

        let sum = CacheBlockedOps::blocked_sum_f64(&data, &config);
        assert_eq!(sum, 500500.0);
    }

    #[test]
    fn test_blocked_sum_prefetch() {
        let config = CacheBlockConfig::default();
        let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();

        let sum = CacheBlockedOps::blocked_sum_f64_prefetch(&data, &config);
        assert_eq!(sum, 500500.0);
    }

    #[test]
    fn test_blocked_min_max() {
        let config = CacheBlockConfig::default();
        let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();

        assert_eq!(CacheBlockedOps::blocked_min_f64(&data, &config), Some(1.0));
        assert_eq!(CacheBlockedOps::blocked_max_f64(&data, &config), Some(1000.0));
    }

    #[test]
    fn test_blocked_dot() {
        let config = CacheBlockConfig::default();
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0, 7.0, 8.0];

        let dot = CacheBlockedOps::blocked_dot_f64(&a, &b, &config);
        assert_eq!(dot, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    }

    #[test]
    fn test_branch_free_ops() {
        assert_eq!(BranchFreeOps::min_f64(3.0, 5.0), 3.0);
        assert_eq!(BranchFreeOps::max_f64(3.0, 5.0), 5.0);
        assert_eq!(BranchFreeOps::clamp_f64(10.0, 0.0, 5.0), 5.0);
        assert_eq!(BranchFreeOps::abs_f64(-5.0), 5.0);
    }

    #[test]
    fn test_conditional_ops() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sum = BranchFreeOps::conditional_sum_f64(&data, 2.5);
        assert_eq!(sum, 12.0); // 3 + 4 + 5

        let count = BranchFreeOps::conditional_count_f64(&data, 2.5);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_streaming_sum() {
        let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let sum = StreamingOps::streaming_sum_f64(&data);
        assert_eq!(sum, 500500.0);
    }

    #[test]
    fn test_tiled_matvec() {
        // 2x3 matrix * 3x1 vector
        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];

        TiledOps::tiled_matvec_f64(&matrix, &vector, &mut result, 2, 3, 2);

        assert_eq!(result[0], 14.0); // 1*1 + 2*2 + 3*3
        assert_eq!(result[1], 32.0); // 4*1 + 5*2 + 6*3
    }

    #[test]
    fn test_tiled_reduction() {
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];

        let row_sums = TiledOps::tiled_reduction_f64(&data, 2, 3, 2);
        assert_eq!(row_sums[0], 6.0);  // 1 + 2 + 3
        assert_eq!(row_sums[1], 15.0); // 4 + 5 + 6
    }

    #[test]
    fn test_prefetcher() {
        let distance = Prefetcher::calculate_distance::<f64>(AccessPattern::Sequential);
        assert!(distance > 0);

        let chunk_size = Prefetcher::optimal_chunk_size::<f64>();
        assert!(chunk_size > 0);
    }

    #[test]
    fn test_bandwidth_optimizer() {
        let optimizer = BandwidthOptimizer::new(50.0);

        let time = optimizer.estimate_time(50_000_000_000); // 50GB
        assert!((time - 1.0).abs() < 0.01);

        assert!(optimizer.is_memory_bound(1_000_000, 100_000)); // Low intensity
        assert!(!optimizer.is_memory_bound(1000, 1_000_000)); // High intensity
    }

    #[test]
    fn test_auto_tuner() {
        let mut tuner = AutoTuner::new();

        assert!(tuner.optimal_block_size(100) <= 100);
        assert!(tuner.should_parallelize(1_000_000, 10));
        assert!(!tuner.should_parallelize(100, 10));

        tuner.record("sum", 1000, 0.001);
        tuner.record("sum", 2000, 0.002);

        let throughput = tuner.average_throughput("sum");
        assert!(throughput.is_some());
        assert!((throughput.unwrap() - 1_000_000.0).abs() < 1.0);
    }
}
