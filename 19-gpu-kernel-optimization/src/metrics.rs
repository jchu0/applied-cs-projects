//! Performance metrics and benchmarking utilities.

use crate::matrix::Matrix;
use crate::Result;
use std::time::{Duration, Instant};

/// Performance metrics for GEMM operations.
#[derive(Debug, Clone)]
pub struct GemmMetrics {
    /// Execution time.
    pub execution_time: Duration,
    /// Matrix dimensions (M, N, K).
    pub dimensions: (usize, usize, usize),
    /// GFLOPS achieved.
    pub gflops: f64,
    /// Memory bandwidth (GB/s).
    pub bandwidth_gbs: f64,
    /// Arithmetic intensity (FLOP/byte).
    pub arithmetic_intensity: f64,
    /// Percentage of peak performance.
    pub efficiency: f64,
}

impl GemmMetrics {
    /// Calculate metrics from execution.
    pub fn calculate(
        m: usize,
        n: usize,
        k: usize,
        execution_time: Duration,
        peak_gflops: f64,
    ) -> Self {
        let time_secs = execution_time.as_secs_f64();

        // Total floating point operations: 2 * M * N * K (multiply-add)
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (time_secs * 1e9);

        // Memory operations: read A (M*K) + read B (K*N) + write C (M*N)
        // Each element is 4 bytes (f32)
        let bytes_accessed = ((m * k + k * n + m * n) * 4) as f64;
        let bandwidth_gbs = bytes_accessed / (time_secs * 1e9);

        // Arithmetic intensity: FLOP per byte accessed
        let arithmetic_intensity = flops / bytes_accessed;

        // Efficiency as percentage of peak
        let efficiency = (gflops / peak_gflops) * 100.0;

        Self {
            execution_time,
            dimensions: (m, n, k),
            gflops,
            bandwidth_gbs,
            arithmetic_intensity,
            efficiency,
        }
    }

    /// Format metrics as human-readable string.
    pub fn format(&self) -> String {
        format!(
            "GEMM {}x{}x{}: {:.2}ms, {:.2} GFLOPS, {:.2} GB/s, AI={:.2}, {:.1}% efficiency",
            self.dimensions.0,
            self.dimensions.1,
            self.dimensions.2,
            self.execution_time.as_secs_f64() * 1000.0,
            self.gflops,
            self.bandwidth_gbs,
            self.arithmetic_intensity,
            self.efficiency
        )
    }
}

/// Benchmark runner for GEMM kernels.
pub struct Benchmark {
    /// Number of warmup iterations.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Estimated peak GFLOPS for the system.
    pub peak_gflops: f64,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self {
            warmup_iters: 3,
            measure_iters: 10,
            // Conservative estimate for modern CPU
            peak_gflops: 100.0,
        }
    }
}

impl Benchmark {
    /// Create new benchmark runner.
    pub fn new(warmup_iters: usize, measure_iters: usize, peak_gflops: f64) -> Self {
        Self {
            warmup_iters,
            measure_iters,
            peak_gflops,
        }
    }

    /// Run benchmark with given kernel function.
    pub fn run<F>(
        &self,
        a: &Matrix,
        b: &Matrix,
        kernel: F,
    ) -> Result<GemmMetrics>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix) -> Result<()>,
    {
        let m = a.rows;
        let n = b.cols;
        let k = a.cols;

        let mut c = Matrix::zeros(m, n);

        // Warmup
        for _ in 0..self.warmup_iters {
            kernel(a, b, &mut c)?;
        }

        // Measure
        let mut times = Vec::with_capacity(self.measure_iters);
        for _ in 0..self.measure_iters {
            let start = Instant::now();
            kernel(a, b, &mut c)?;
            times.push(start.elapsed());
        }

        // Use median time to avoid outliers
        times.sort();
        let median_time = times[times.len() / 2];

        Ok(GemmMetrics::calculate(m, n, k, median_time, self.peak_gflops))
    }

    /// Run benchmark and return all timing results.
    pub fn run_detailed<F>(
        &self,
        a: &Matrix,
        b: &Matrix,
        kernel: F,
    ) -> Result<BenchmarkResults>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix) -> Result<()>,
    {
        let m = a.rows;
        let n = b.cols;
        let k = a.cols;

        let mut c = Matrix::zeros(m, n);

        // Warmup
        for _ in 0..self.warmup_iters {
            kernel(a, b, &mut c)?;
        }

        // Measure
        let mut times = Vec::with_capacity(self.measure_iters);
        for _ in 0..self.measure_iters {
            let start = Instant::now();
            kernel(a, b, &mut c)?;
            times.push(start.elapsed());
        }

        Ok(BenchmarkResults::new(m, n, k, times, self.peak_gflops))
    }
}

/// Detailed benchmark results with statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Matrix dimensions.
    pub dimensions: (usize, usize, usize),
    /// All timing measurements.
    pub times: Vec<Duration>,
    /// Minimum time.
    pub min_time: Duration,
    /// Maximum time.
    pub max_time: Duration,
    /// Median time.
    pub median_time: Duration,
    /// Mean time.
    pub mean_time: Duration,
    /// Standard deviation.
    pub std_dev: Duration,
    /// Best GFLOPS achieved.
    pub best_gflops: f64,
    /// Median GFLOPS.
    pub median_gflops: f64,
}

impl BenchmarkResults {
    /// Create results from timing data.
    pub fn new(
        m: usize,
        n: usize,
        k: usize,
        mut times: Vec<Duration>,
        peak_gflops: f64,
    ) -> Self {
        times.sort();

        let min_time = times[0];
        let max_time = times[times.len() - 1];
        let median_time = times[times.len() / 2];

        let sum: Duration = times.iter().sum();
        let mean_time = sum / times.len() as u32;

        // Calculate standard deviation
        let mean_nanos = mean_time.as_nanos() as f64;
        let variance: f64 = times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let best_gflops = flops / (min_time.as_secs_f64() * 1e9);
        let median_gflops = flops / (median_time.as_secs_f64() * 1e9);

        Self {
            dimensions: (m, n, k),
            times,
            min_time,
            max_time,
            median_time,
            mean_time,
            std_dev,
            best_gflops,
            median_gflops,
        }
    }

    /// Get coefficient of variation (CV).
    pub fn cv(&self) -> f64 {
        self.std_dev.as_nanos() as f64 / self.mean_time.as_nanos() as f64
    }

    /// Format results as string.
    pub fn format(&self) -> String {
        format!(
            "GEMM {}x{}x{}:\n\
             \x20 Best:   {:.2}ms ({:.2} GFLOPS)\n\
             \x20 Median: {:.2}ms ({:.2} GFLOPS)\n\
             \x20 Mean:   {:.2}ms ± {:.2}ms (CV={:.1}%)",
            self.dimensions.0,
            self.dimensions.1,
            self.dimensions.2,
            self.min_time.as_secs_f64() * 1000.0,
            self.best_gflops,
            self.median_time.as_secs_f64() * 1000.0,
            self.median_gflops,
            self.mean_time.as_secs_f64() * 1000.0,
            self.std_dev.as_secs_f64() * 1000.0,
            self.cv() * 100.0
        )
    }
}

/// Performance model for predicting GEMM performance.
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// Memory bandwidth (GB/s).
    pub memory_bandwidth: f64,
    /// Peak compute (GFLOPS).
    pub peak_compute: f64,
    /// Cache sizes (L1, L2, L3 in bytes).
    pub cache_sizes: (usize, usize, usize),
}

impl Default for PerformanceModel {
    fn default() -> Self {
        Self {
            memory_bandwidth: 50.0,  // 50 GB/s typical DDR4
            peak_compute: 100.0,     // 100 GFLOPS typical modern CPU
            cache_sizes: (32 * 1024, 256 * 1024, 8 * 1024 * 1024),
        }
    }
}

impl PerformanceModel {
    /// Create roofline model.
    pub fn new(memory_bandwidth: f64, peak_compute: f64) -> Self {
        Self {
            memory_bandwidth,
            peak_compute,
            cache_sizes: (32 * 1024, 256 * 1024, 8 * 1024 * 1024),
        }
    }

    /// Predict maximum achievable GFLOPS given arithmetic intensity.
    pub fn predict_gflops(&self, arithmetic_intensity: f64) -> f64 {
        // Roofline model: min(peak, bandwidth * intensity)
        let memory_bound = self.memory_bandwidth * arithmetic_intensity;
        memory_bound.min(self.peak_compute)
    }

    /// Calculate arithmetic intensity for GEMM.
    pub fn gemm_arithmetic_intensity(m: usize, n: usize, k: usize) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = ((m * k + k * n + m * n) * 4) as f64;
        flops / bytes
    }

    /// Predict GEMM performance.
    pub fn predict_gemm(&self, m: usize, n: usize, k: usize) -> f64 {
        let intensity = Self::gemm_arithmetic_intensity(m, n, k);
        self.predict_gflops(intensity)
    }

    /// Check if operation is memory-bound or compute-bound.
    pub fn is_memory_bound(&self, arithmetic_intensity: f64) -> bool {
        let ridge_point = self.peak_compute / self.memory_bandwidth;
        arithmetic_intensity < ridge_point
    }

    /// Get ridge point (where memory-bound meets compute-bound).
    pub fn ridge_point(&self) -> f64 {
        self.peak_compute / self.memory_bandwidth
    }

    /// Estimate optimal tile size for cache.
    pub fn optimal_tile_size(&self, cache_level: usize) -> usize {
        let cache_size = match cache_level {
            1 => self.cache_sizes.0,
            2 => self.cache_sizes.1,
            3 => self.cache_sizes.2,
            _ => self.cache_sizes.1,
        };

        // For tiled GEMM, we need 3 tile matrices to fit in cache
        // tile_size^2 * 3 * sizeof(f32) <= cache_size
        let elements_per_tile = cache_size / 12; // 3 matrices * 4 bytes
        (elements_per_tile as f64).sqrt() as usize
    }
}

/// Compare performance of multiple kernels.
pub struct KernelComparison {
    /// Results for each kernel.
    pub results: Vec<(String, BenchmarkResults)>,
    /// Baseline kernel name.
    pub baseline: String,
}

impl KernelComparison {
    /// Create new comparison.
    pub fn new(baseline: String) -> Self {
        Self {
            results: Vec::new(),
            baseline,
        }
    }

    /// Add kernel results.
    pub fn add(&mut self, name: String, results: BenchmarkResults) {
        self.results.push((name, results));
    }

    /// Get speedup compared to baseline.
    pub fn speedup(&self, kernel: &str) -> Option<f64> {
        let baseline_time = self.results
            .iter()
            .find(|(n, _)| n == &self.baseline)
            .map(|(_, r)| r.median_time)?;

        let kernel_time = self.results
            .iter()
            .find(|(n, _)| n == kernel)
            .map(|(_, r)| r.median_time)?;

        Some(baseline_time.as_secs_f64() / kernel_time.as_secs_f64())
    }

    /// Format comparison as table.
    pub fn format(&self) -> String {
        let mut output = String::from("Kernel Comparison:\n");
        output.push_str(&format!(
            "{:<20} {:>10} {:>10} {:>10}\n",
            "Kernel", "Time (ms)", "GFLOPS", "Speedup"
        ));
        output.push_str(&"-".repeat(52));
        output.push('\n');

        let baseline_time = self.results
            .iter()
            .find(|(n, _)| n == &self.baseline)
            .map(|(_, r)| r.median_time)
            .unwrap_or(Duration::from_secs(1));

        for (name, results) in &self.results {
            let speedup = baseline_time.as_secs_f64() / results.median_time.as_secs_f64();
            output.push_str(&format!(
                "{:<20} {:>10.2} {:>10.2} {:>10.2}x\n",
                name,
                results.median_time.as_secs_f64() * 1000.0,
                results.median_gflops,
                speedup
            ));
        }

        output
    }
}

/// Profile memory access patterns (simulated).
#[derive(Debug, Clone, Default)]
pub struct MemoryProfile {
    /// Total memory accesses.
    pub total_accesses: usize,
    /// Sequential accesses.
    pub sequential_accesses: usize,
    /// Strided accesses.
    pub strided_accesses: usize,
    /// Random accesses.
    pub random_accesses: usize,
    /// Cache hits (estimated).
    pub cache_hits: usize,
    /// Cache misses (estimated).
    pub cache_misses: usize,
}

impl MemoryProfile {
    /// Calculate cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_accesses as f64
        }
    }

    /// Profile naive GEMM access pattern.
    pub fn profile_naive_gemm(m: usize, n: usize, k: usize, cache_line_size: usize) -> Self {
        let elements_per_line = cache_line_size / 4; // f32

        // Each output element accesses: K elements of A (sequential) + K elements of B (strided)
        let total_accesses = m * n * k * 2;
        let sequential_accesses = m * n * k; // A accesses
        let strided_accesses = m * n * k;    // B accesses

        // Estimate cache behavior
        // A: good temporal locality within row
        // B: poor locality due to column access
        let cache_hits = sequential_accesses - (sequential_accesses / elements_per_line);
        let cache_misses = total_accesses - cache_hits;

        Self {
            total_accesses,
            sequential_accesses,
            strided_accesses,
            random_accesses: 0,
            cache_hits,
            cache_misses,
        }
    }

    /// Profile tiled GEMM access pattern.
    pub fn profile_tiled_gemm(
        m: usize,
        n: usize,
        k: usize,
        tile_size: usize,
        cache_line_size: usize,
    ) -> Self {
        let elements_per_line = cache_line_size / 4;

        let num_tiles_m = (m + tile_size - 1) / tile_size;
        let num_tiles_n = (n + tile_size - 1) / tile_size;
        let num_tiles_k = (k + tile_size - 1) / tile_size;

        // Each tile pair loads tile_size^2 elements
        let tile_loads = num_tiles_m * num_tiles_n * num_tiles_k * 2 * tile_size * tile_size;
        let compute_accesses = num_tiles_m * num_tiles_n * num_tiles_k * tile_size * tile_size * tile_size * 2;

        let total_accesses = tile_loads + compute_accesses;

        // Better locality from tiling
        let cache_hits = compute_accesses - (compute_accesses / (tile_size * tile_size));
        let cache_misses = total_accesses - cache_hits;

        Self {
            total_accesses,
            sequential_accesses: tile_loads,
            strided_accesses: 0,
            random_accesses: 0,
            cache_hits,
            cache_misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_metrics() {
        let metrics = GemmMetrics::calculate(
            1024, 1024, 1024,
            Duration::from_millis(100),
            100.0,
        );

        // 2 * 1024^3 = 2^31 FLOP
        // In 100ms = 0.1s
        // GFLOPS = 2^31 / (0.1 * 10^9) ≈ 21.5
        assert!(metrics.gflops > 20.0 && metrics.gflops < 25.0);
    }

    #[test]
    fn test_performance_model() {
        let model = PerformanceModel::new(50.0, 100.0);

        // Ridge point = 100 / 50 = 2 FLOP/byte
        assert!((model.ridge_point() - 2.0).abs() < 0.01);

        // Memory bound case
        assert!(model.is_memory_bound(1.0));
        assert_eq!(model.predict_gflops(1.0), 50.0);

        // Compute bound case
        assert!(!model.is_memory_bound(3.0));
        assert_eq!(model.predict_gflops(3.0), 100.0);
    }

    #[test]
    fn test_benchmark_results() {
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
    }
}
