//! Auto-selection engine for optimal attention algorithm.

use crate::attention::{AttentionOutput, StandardAttention};
use crate::block_sparse::BlockSparseAttention;
use crate::config::{AttentionConfig, AttentionType, TensorShape};
use crate::flash::FlashAttention;
use crate::sliding_window::SlidingWindowAttention;
use crate::Result;

/// GPU architecture for tuning thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArchitecture {
    /// NVIDIA V100 (Volta).
    V100,
    /// NVIDIA A100 (Ampere).
    A100,
    /// NVIDIA H100 (Hopper).
    H100,
    /// Generic/unknown GPU.
    Generic,
}

impl Default for GpuArchitecture {
    fn default() -> Self {
        Self::Generic
    }
}

/// Thresholds for algorithm selection.
#[derive(Debug, Clone)]
pub struct SelectionThresholds {
    /// Minimum sequence length for FlashAttention.
    pub flash_min: usize,
    /// Minimum sequence length for sliding window.
    pub sliding_window_min: usize,
    /// Minimum sequence length for block sparse.
    pub block_sparse_min: usize,
}

impl SelectionThresholds {
    /// Get thresholds for GPU architecture.
    pub fn for_gpu(arch: GpuArchitecture) -> Self {
        match arch {
            GpuArchitecture::A100 => Self {
                flash_min: 256,
                sliding_window_min: 8192,
                block_sparse_min: 32768,
            },
            GpuArchitecture::H100 => Self {
                flash_min: 128,
                sliding_window_min: 16384,
                block_sparse_min: 65536,
            },
            GpuArchitecture::V100 | GpuArchitecture::Generic => Self {
                flash_min: 512,
                sliding_window_min: 4096,
                block_sparse_min: 16384,
            },
        }
    }
}

impl Default for SelectionThresholds {
    fn default() -> Self {
        Self::for_gpu(GpuArchitecture::Generic)
    }
}

/// Auto-select attention implementation.
pub struct AutoSelectAttention {
    config: AttentionConfig,
    thresholds: SelectionThresholds,
    /// Standard attention for short sequences.
    standard: StandardAttention,
    /// Flash attention for medium sequences.
    flash: FlashAttention,
    /// Sliding window for long sequences.
    sliding_window: SlidingWindowAttention,
    /// Block sparse for very long sequences.
    block_sparse: BlockSparseAttention,
}

impl AutoSelectAttention {
    /// Create new auto-select attention.
    pub fn new(config: AttentionConfig) -> Self {
        Self::with_thresholds(config.clone(), SelectionThresholds::default())
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(config: AttentionConfig, thresholds: SelectionThresholds) -> Self {
        Self {
            standard: StandardAttention::new(config.clone()),
            flash: FlashAttention::new(config.clone()),
            sliding_window: SlidingWindowAttention::new(config.clone()),
            block_sparse: BlockSparseAttention::new(config.clone()),
            config,
            thresholds,
        }
    }

    /// Create with GPU-specific thresholds.
    pub fn for_gpu(config: AttentionConfig, arch: GpuArchitecture) -> Self {
        Self::with_thresholds(config, SelectionThresholds::for_gpu(arch))
    }

    /// Select optimal implementation for given sequence length.
    pub fn select_impl(&self, seq_len: usize) -> AttentionType {
        if seq_len < self.thresholds.flash_min {
            AttentionType::Standard
        } else if seq_len < self.thresholds.sliding_window_min {
            AttentionType::Flash
        } else if seq_len < self.thresholds.block_sparse_min {
            AttentionType::SlidingWindow
        } else {
            AttentionType::BlockSparse
        }
    }

    /// Compute attention with automatic algorithm selection.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput> {
        let impl_type = self.select_impl(q_shape.seq_len);

        match impl_type {
            AttentionType::Standard => {
                self.standard.forward(query, key, value, q_shape, kv_shape, attention_mask)
            }
            AttentionType::Flash => {
                self.flash.forward(query, key, value, q_shape, kv_shape, attention_mask)
            }
            AttentionType::SlidingWindow => {
                self.sliding_window.forward(query, key, value, q_shape, kv_shape, attention_mask)
            }
            AttentionType::BlockSparse => {
                self.block_sparse.forward(query, key, value, q_shape, kv_shape, attention_mask)
            }
            _ => {
                // Fallback to standard
                self.standard.forward(query, key, value, q_shape, kv_shape, attention_mask)
            }
        }
    }

    /// Estimate memory usage for given parameters.
    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize {
        let impl_type = self.select_impl(seq_len);

        match impl_type {
            AttentionType::Standard => {
                // O(n^2) for attention matrix
                let num_heads = self.config.num_heads;
                let head_dim = self.config.head_dim;
                batch * num_heads * seq_len * seq_len * 4
                    + batch * seq_len * num_heads * head_dim * 4
            }
            AttentionType::Flash => self.flash.estimate_memory(batch, seq_len),
            AttentionType::SlidingWindow => self.sliding_window.estimate_memory(batch, seq_len),
            AttentionType::BlockSparse => self.block_sparse.estimate_memory(batch, seq_len),
            _ => 0,
        }
    }
}

/// Performance profiler for attention operations.
pub struct AttentionProfiler {
    /// Recorded timings: (seq_len, impl_type, time_ms).
    timings: Vec<(usize, AttentionType, f64)>,
}

impl AttentionProfiler {
    /// Create new profiler.
    pub fn new() -> Self {
        Self {
            timings: Vec::new(),
        }
    }

    /// Record a timing.
    pub fn record(&mut self, seq_len: usize, impl_type: AttentionType, time_ms: f64) {
        self.timings.push((seq_len, impl_type, time_ms));
    }

    /// Get average time for an implementation at a sequence length.
    pub fn average_time(&self, seq_len: usize, impl_type: AttentionType) -> Option<f64> {
        let relevant: Vec<f64> = self
            .timings
            .iter()
            .filter(|(s, t, _)| *s == seq_len && *t == impl_type)
            .map(|(_, _, time)| *time)
            .collect();

        if relevant.is_empty() {
            None
        } else {
            Some(relevant.iter().sum::<f64>() / relevant.len() as f64)
        }
    }

    /// Get best implementation for a sequence length based on timings.
    pub fn best_impl(&self, seq_len: usize) -> Option<AttentionType> {
        let impl_types = [
            AttentionType::Standard,
            AttentionType::Flash,
            AttentionType::SlidingWindow,
            AttentionType::BlockSparse,
        ];

        let mut best: Option<(AttentionType, f64)> = None;

        for &impl_type in &impl_types {
            if let Some(time) = self.average_time(seq_len, impl_type) {
                if best.map(|(_, t)| time < t).unwrap_or(true) {
                    best = Some((impl_type, time));
                }
            }
        }

        best.map(|(t, _)| t)
    }

    /// Clear recorded timings.
    pub fn clear(&mut self) {
        self.timings.clear();
    }

    /// Get all timings.
    pub fn timings(&self) -> &[(usize, AttentionType, f64)] {
        &self.timings
    }
}

impl Default for AttentionProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory estimator for different attention configurations.
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory for standard attention (O(n^2)).
    pub fn standard(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> usize {
        // Attention scores: batch * heads * seq^2
        // Output: batch * seq * heads * dim
        let scores = batch * num_heads * seq_len * seq_len * 4;
        let output = batch * seq_len * num_heads * head_dim * 4;
        scores + output
    }

    /// Estimate memory for flash attention (O(n)).
    pub fn flash(
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> usize {
        // Tile memory: block_size^2
        // Output: batch * seq * heads * dim
        // Logsumexp: batch * heads * seq
        let tile = block_size * block_size * 4;
        let output = batch * seq_len * num_heads * head_dim * 4;
        let lse = batch * num_heads * seq_len * 4;
        tile + output + lse
    }

    /// Estimate memory for sliding window attention.
    pub fn sliding_window(
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        window_size: usize,
    ) -> usize {
        let effective_window = window_size.min(seq_len);
        let scores = batch * num_heads * seq_len * effective_window * 4;
        let output = batch * seq_len * num_heads * head_dim * 4;
        scores + output
    }

    /// Estimate total transformer memory.
    pub fn transformer(
        batch: usize,
        seq_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> usize {
        // Per layer: attention + FFN
        let attention = Self::standard(batch, seq_len, num_heads, head_dim);
        let ffn = batch * seq_len * intermediate_dim * 4 * 2; // Two matmuls
        let per_layer = attention + ffn;

        // KV cache
        let kv_cache = 2 * batch * seq_len * num_heads * head_dim * 4 * num_layers;

        per_layer * num_layers + kv_cache
    }

    /// Check if configuration fits in memory.
    pub fn fits_in_memory(
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        available_memory: usize,
    ) -> bool {
        Self::standard(batch, seq_len, num_heads, head_dim) < available_memory
    }

    /// Find maximum sequence length that fits in memory.
    pub fn max_seq_len(
        batch: usize,
        num_heads: usize,
        _head_dim: usize,
        available_memory: usize,
    ) -> usize {
        // Solve: batch * heads * seq^2 * 4 <= available
        // seq^2 <= available / (batch * heads * 4)
        let max_scores = available_memory / (batch * num_heads * 4);
        (max_scores as f64).sqrt() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_select() {
        let config = AttentionConfig::new(8, 64);
        let auto = AutoSelectAttention::new(config);

        // Test selection thresholds
        assert_eq!(auto.select_impl(100), AttentionType::Standard);
        assert_eq!(auto.select_impl(1000), AttentionType::Flash);
        assert_eq!(auto.select_impl(8000), AttentionType::SlidingWindow);
        assert_eq!(auto.select_impl(20000), AttentionType::BlockSparse);
    }

    #[test]
    fn test_auto_forward() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let auto = AutoSelectAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = auto.forward(&query, &key, &value, shape, shape, None).unwrap();
        assert_eq!(output.output.len(), size);
    }

    #[test]
    fn test_auto_forward_short_buffer_errors() {
        // Auto-select delegates to an underlying impl; validation must still
        // surface a clean error rather than panicking.
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let auto = AutoSelectAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.01; size - 2]; // too short
        let key: Vec<f32> = vec![0.01; size];
        let value: Vec<f32> = vec![0.01; size];

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = auto.forward(&query, &key, &value, shape, shape, None);

        assert!(matches!(
            result,
            Err(crate::Error::ShapeMismatch { name: "query", .. })
        ));
    }

    #[test]
    fn test_gpu_thresholds() {
        let h100 = SelectionThresholds::for_gpu(GpuArchitecture::H100);
        let v100 = SelectionThresholds::for_gpu(GpuArchitecture::V100);

        // H100 should have higher thresholds (better hardware)
        assert!(h100.sliding_window_min > v100.sliding_window_min);
        assert!(h100.block_sparse_min > v100.block_sparse_min);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = AttentionProfiler::new();

        profiler.record(1024, AttentionType::Standard, 5.0);
        profiler.record(1024, AttentionType::Standard, 5.5);
        profiler.record(1024, AttentionType::Flash, 3.0);

        let avg = profiler.average_time(1024, AttentionType::Standard).unwrap();
        assert!((avg - 5.25).abs() < 0.01);

        let best = profiler.best_impl(1024).unwrap();
        assert_eq!(best, AttentionType::Flash);
    }

    #[test]
    fn test_memory_estimator() {
        // Standard attention: O(n^2)
        let standard = MemoryEstimator::standard(1, 16384, 32, 128);
        // Flash attention: O(n)
        let flash = MemoryEstimator::flash(1, 16384, 32, 128, 64);

        // Flash should use much less memory
        assert!(flash < standard / 10);

        // Test max sequence length
        let max_seq = MemoryEstimator::max_seq_len(1, 32, 128, 16 * 1024 * 1024 * 1024);
        assert!(max_seq > 10000); // Should support >10K with 16GB
    }

    #[test]
    fn test_sliding_window_memory() {
        let full = MemoryEstimator::standard(1, 32768, 32, 128);
        let window = MemoryEstimator::sliding_window(1, 32768, 32, 128, 4096);

        // Window should use ~8x less memory (4096/32768)
        assert!(window < full / 4);
    }
}
