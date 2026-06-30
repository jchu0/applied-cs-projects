//! Performance metrics and monitoring.

use std::time::{Duration, Instant};

/// Performance metrics for a single operation.
#[derive(Debug, Clone)]
pub struct PerfMetrics {
    /// Total execution time.
    pub execution_time: Duration,
    /// Number of rows processed.
    pub rows_processed: usize,
    /// Number of bytes processed.
    pub bytes_processed: usize,
    /// Throughput in rows/second.
    pub rows_per_second: f64,
    /// Throughput in GB/second.
    pub gb_per_second: f64,
}

impl PerfMetrics {
    /// Calculate metrics from execution data.
    pub fn calculate(
        execution_time: Duration,
        rows_processed: usize,
        bytes_processed: usize,
    ) -> Self {
        let secs = execution_time.as_secs_f64();
        let rows_per_second = if secs > 0.0 {
            rows_processed as f64 / secs
        } else {
            0.0
        };
        let gb_per_second = if secs > 0.0 {
            (bytes_processed as f64 / 1e9) / secs
        } else {
            0.0
        };

        Self {
            execution_time,
            rows_processed,
            bytes_processed,
            rows_per_second,
            gb_per_second,
        }
    }

    /// Format metrics for display.
    pub fn format(&self) -> String {
        format!(
            "Time: {:.2}ms | Rows: {} | {:.2} Mrows/s | {:.2} GB/s",
            self.execution_time.as_secs_f64() * 1000.0,
            self.rows_processed,
            self.rows_per_second / 1e6,
            self.gb_per_second
        )
    }
}

/// Query execution metrics.
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Total query time.
    pub total_time: Duration,
    /// Scan time.
    pub scan_time: Duration,
    /// Filter time.
    pub filter_time: Duration,
    /// Aggregate time.
    pub aggregate_time: Duration,
    /// Sort time.
    pub sort_time: Duration,
    /// Number of rows scanned.
    pub rows_scanned: usize,
    /// Number of rows after filter.
    pub rows_filtered: usize,
    /// Number of output rows.
    pub rows_output: usize,
    /// Bytes scanned.
    pub bytes_scanned: usize,
}

impl QueryMetrics {
    /// Create new query metrics.
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            scan_time: Duration::ZERO,
            filter_time: Duration::ZERO,
            aggregate_time: Duration::ZERO,
            sort_time: Duration::ZERO,
            rows_scanned: 0,
            rows_filtered: 0,
            rows_output: 0,
            bytes_scanned: 0,
        }
    }

    /// Calculate throughput.
    pub fn throughput(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.rows_scanned as f64 / secs
        } else {
            0.0
        }
    }

    /// Calculate selectivity.
    pub fn selectivity(&self) -> f64 {
        if self.rows_scanned > 0 {
            self.rows_filtered as f64 / self.rows_scanned as f64
        } else {
            0.0
        }
    }

    /// Format as dashboard.
    pub fn format_dashboard(&self) -> String {
        let mut output = String::new();
        output.push_str("╔══════════════════════════════════════════════╗\n");
        output.push_str("║       Query Performance Dashboard            ║\n");
        output.push_str("╠══════════════════════════════════════════════╣\n");
        output.push_str(&format!(
            "║ Total Time:     {:>10.2} ms              ║\n",
            self.total_time.as_secs_f64() * 1000.0
        ));
        output.push_str(&format!(
            "║ Scan Time:      {:>10.2} ms              ║\n",
            self.scan_time.as_secs_f64() * 1000.0
        ));
        output.push_str(&format!(
            "║ Filter Time:    {:>10.2} ms              ║\n",
            self.filter_time.as_secs_f64() * 1000.0
        ));
        output.push_str(&format!(
            "║ Aggregate Time: {:>10.2} ms              ║\n",
            self.aggregate_time.as_secs_f64() * 1000.0
        ));
        output.push_str("║                                              ║\n");
        output.push_str(&format!(
            "║ Rows Scanned:   {:>10}                  ║\n",
            self.rows_scanned
        ));
        output.push_str(&format!(
            "║ Rows Filtered:  {:>10}                  ║\n",
            self.rows_filtered
        ));
        output.push_str(&format!(
            "║ Selectivity:    {:>10.1}%                 ║\n",
            self.selectivity() * 100.0
        ));
        output.push_str("║                                              ║\n");
        output.push_str(&format!(
            "║ Throughput:     {:>10.2} M rows/s        ║\n",
            self.throughput() / 1e6
        ));
        output.push_str(&format!(
            "║ Bandwidth:      {:>10.2} GB/s            ║\n",
            (self.bytes_scanned as f64 / 1e9) / self.total_time.as_secs_f64()
        ));
        output.push_str("╚══════════════════════════════════════════════╝\n");
        output
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation duration.
pub struct Timer {
    start: Instant,
    splits: Vec<(String, Duration)>,
}

impl Timer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            splits: Vec::new(),
        }
    }

    /// Record a split time.
    pub fn split(&mut self, name: impl Into<String>) {
        let elapsed = self.start.elapsed();
        let prev_total: Duration = self.splits.iter().map(|(_, d)| *d).sum();
        let split_time = elapsed - prev_total;
        self.splits.push((name.into(), split_time));
    }

    /// Get total elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get split times.
    pub fn splits(&self) -> &[(String, Duration)] {
        &self.splits
    }

    /// Format splits for display.
    pub fn format_splits(&self) -> String {
        let mut output = String::new();
        for (name, duration) in &self.splits {
            output.push_str(&format!(
                "{}: {:.2}ms\n",
                name,
                duration.as_secs_f64() * 1000.0
            ));
        }
        output
    }
}

/// Benchmark runner for performance testing.
pub struct Benchmark {
    /// Number of warmup iterations.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self {
            warmup_iters: 3,
            measure_iters: 10,
        }
    }
}

impl Benchmark {
    /// Create new benchmark.
    pub fn new(warmup_iters: usize, measure_iters: usize) -> Self {
        Self {
            warmup_iters,
            measure_iters,
        }
    }

    /// Run benchmark.
    pub fn run<F>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iters {
            f();
        }

        // Measure
        let mut times = Vec::with_capacity(self.measure_iters);
        for _ in 0..self.measure_iters {
            let start = Instant::now();
            f();
            times.push(start.elapsed());
        }

        BenchmarkResult::from_times(times)
    }

    /// Run benchmark with setup.
    pub fn run_with_setup<S, F, T>(&self, setup: S, mut f: F) -> BenchmarkResult
    where
        S: Fn() -> T,
        F: FnMut(T),
    {
        // Warmup
        for _ in 0..self.warmup_iters {
            let input = setup();
            f(input);
        }

        // Measure
        let mut times = Vec::with_capacity(self.measure_iters);
        for _ in 0..self.measure_iters {
            let input = setup();
            let start = Instant::now();
            f(input);
            times.push(start.elapsed());
        }

        BenchmarkResult::from_times(times)
    }
}

/// Benchmark result with statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
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
}

impl BenchmarkResult {
    /// Create result from timing data.
    pub fn from_times(mut times: Vec<Duration>) -> Self {
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

        Self {
            times,
            min_time,
            max_time,
            median_time,
            mean_time,
            std_dev,
        }
    }

    /// Format result for display.
    pub fn format(&self) -> String {
        format!(
            "min: {:.2}ms | median: {:.2}ms | mean: {:.2}ms ± {:.2}ms | max: {:.2}ms",
            self.min_time.as_secs_f64() * 1000.0,
            self.median_time.as_secs_f64() * 1000.0,
            self.mean_time.as_secs_f64() * 1000.0,
            self.std_dev.as_secs_f64() * 1000.0,
            self.max_time.as_secs_f64() * 1000.0
        )
    }
}

/// Cache statistics (estimated).
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Estimated L1 hits.
    pub l1_hits: usize,
    /// Estimated L1 misses.
    pub l1_misses: usize,
    /// Estimated L2 hits.
    pub l2_hits: usize,
    /// Estimated L2 misses.
    pub l2_misses: usize,
    /// Estimated L3 hits.
    pub l3_hits: usize,
    /// Estimated L3 misses.
    pub l3_misses: usize,
}

impl CacheStats {
    /// Estimate cache behavior for sequential scan.
    pub fn estimate_sequential_scan(bytes: usize, l1_size: usize, l2_size: usize, l3_size: usize) -> Self {
        let cache_line_size = 64;
        let num_lines = bytes / cache_line_size;

        let l1_capacity = l1_size / cache_line_size;
        let l2_capacity = l2_size / cache_line_size;
        let l3_capacity = l3_size / cache_line_size;

        // For sequential scan, we get compulsory misses for first access
        // then temporal locality helps within cache capacity
        let l1_hits = if num_lines <= l1_capacity {
            0
        } else {
            (num_lines - l1_capacity).min(num_lines * 7 / 8) // Estimate prefetch benefit
        };
        let l1_misses = num_lines - l1_hits;

        let l2_hits = l1_misses * 3 / 4; // Many L1 misses hit in L2
        let l2_misses = l1_misses - l2_hits;

        let l3_hits = l2_misses * 3 / 4;
        let l3_misses = l2_misses - l3_hits;

        Self {
            l1_hits,
            l1_misses,
            l2_hits,
            l2_misses,
            l3_hits,
            l3_misses,
        }
    }

    /// Calculate L1 hit rate.
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total > 0 {
            self.l1_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate L2 hit rate.
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l2_hits + self.l2_misses;
        if total > 0 {
            self.l2_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate L3 hit rate.
    pub fn l3_hit_rate(&self) -> f64 {
        let total = self.l3_hits + self.l3_misses;
        if total > 0 {
            self.l3_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Performance model for estimating operation cost.
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// Memory bandwidth (GB/s).
    pub memory_bandwidth: f64,
    /// L1 cache size (bytes).
    pub l1_cache_size: usize,
    /// L2 cache size (bytes).
    pub l2_cache_size: usize,
    /// L3 cache size (bytes).
    pub l3_cache_size: usize,
    /// L1 latency (cycles).
    pub l1_latency: f64,
    /// L2 latency (cycles).
    pub l2_latency: f64,
    /// L3 latency (cycles).
    pub l3_latency: f64,
    /// Memory latency (cycles).
    pub mem_latency: f64,
    /// CPU frequency (GHz).
    pub cpu_freq: f64,
}

impl Default for PerformanceModel {
    fn default() -> Self {
        Self {
            memory_bandwidth: 50.0,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
            l1_latency: 4.0,
            l2_latency: 12.0,
            l3_latency: 40.0,
            mem_latency: 100.0,
            cpu_freq: 3.0,
        }
    }
}

impl PerformanceModel {
    /// Estimate time for memory-bound operation.
    pub fn estimate_memory_bound(&self, bytes: usize) -> Duration {
        let secs = (bytes as f64 / 1e9) / self.memory_bandwidth;
        Duration::from_secs_f64(secs)
    }

    /// Estimate time for compute-bound operation.
    pub fn estimate_compute_bound(&self, operations: usize) -> Duration {
        // Assume 8 operations per cycle with SIMD
        let cycles = operations as f64 / 8.0;
        let secs = cycles / (self.cpu_freq * 1e9);
        Duration::from_secs_f64(secs)
    }

    /// Estimate scan performance.
    pub fn estimate_scan(&self, num_rows: usize, row_bytes: usize) -> Duration {
        let bytes = num_rows * row_bytes;
        self.estimate_memory_bound(bytes)
    }

    /// Estimate aggregate performance.
    pub fn estimate_aggregate(&self, num_rows: usize) -> Duration {
        // One operation per row
        let compute = self.estimate_compute_bound(num_rows);
        // Plus memory access
        let memory = self.estimate_memory_bound(num_rows * 8);
        compute + memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_metrics() {
        let metrics = PerfMetrics::calculate(
            Duration::from_millis(100),
            1_000_000,
            8_000_000,
        );

        assert_eq!(metrics.rows_per_second, 10_000_000.0);
        assert!((metrics.gb_per_second - 0.08).abs() < 0.001);
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        timer.split("phase1");
        std::thread::sleep(Duration::from_millis(10));
        timer.split("phase2");

        assert_eq!(timer.splits().len(), 2);
        assert!(timer.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn test_benchmark() {
        let benchmark = Benchmark::new(1, 5);
        let result = benchmark.run(|| {
            std::thread::sleep(Duration::from_millis(1));
        });

        assert_eq!(result.times.len(), 5);
        assert!(result.min_time >= Duration::from_millis(1));
    }

    #[test]
    fn test_benchmark_result() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(11),
            Duration::from_millis(12),
            Duration::from_millis(9),
            Duration::from_millis(13),
        ];

        let result = BenchmarkResult::from_times(times);

        assert_eq!(result.min_time, Duration::from_millis(9));
        assert_eq!(result.max_time, Duration::from_millis(13));
        assert_eq!(result.median_time, Duration::from_millis(11));
    }
}
