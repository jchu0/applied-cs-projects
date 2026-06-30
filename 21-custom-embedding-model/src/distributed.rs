//! Distributed training support with DDP and FSDP.
//!
//! Provides abstractions for:
//! - Distributed Data Parallel (DDP) training
//! - Fully Sharded Data Parallel (FSDP)
//! - Gradient synchronization across workers
//! - Process groups and communication

use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Backend for distributed communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBackend {
    /// NCCL backend for GPU.
    Nccl,
    /// Gloo backend for CPU.
    Gloo,
    /// MPI backend.
    Mpi,
    /// Simulated backend for testing.
    Simulated,
}

impl Default for DistributedBackend {
    fn default() -> Self {
        Self::Simulated
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Communication backend.
    pub backend: DistributedBackend,
    /// World size (total number of processes).
    pub world_size: usize,
    /// Current rank.
    pub rank: usize,
    /// Local rank (GPU index on this node).
    pub local_rank: usize,
    /// Master address for rendezvous.
    pub master_addr: String,
    /// Master port for rendezvous.
    pub master_port: u16,
    /// Timeout for operations.
    pub timeout: Duration,
    /// Use gradient compression.
    pub gradient_compression: bool,
    /// Bucket size for gradient bucketing (MB).
    pub bucket_size_mb: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            backend: DistributedBackend::Simulated,
            world_size: 1,
            rank: 0,
            local_rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            timeout: Duration::from_secs(1800),
            gradient_compression: false,
            bucket_size_mb: 25,
        }
    }
}

impl DistributedConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Result<Self> {
        let world_size = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let rank = std::env::var("RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let master_addr = std::env::var("MASTER_ADDR")
            .unwrap_or_else(|_| "localhost".to_string());

        let master_port = std::env::var("MASTER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(29500);

        Ok(Self {
            world_size,
            rank,
            local_rank,
            master_addr,
            master_port,
            ..Default::default()
        })
    }

    /// Check if this is the main process (rank 0).
    pub fn is_main_process(&self) -> bool {
        self.rank == 0
    }
}

/// Process group for collective communications.
#[derive(Debug)]
pub struct ProcessGroup {
    /// Configuration.
    config: DistributedConfig,
    /// Whether initialized.
    initialized: bool,
    /// Barrier counter for synchronization.
    barrier_counter: Arc<Mutex<usize>>,
}

impl ProcessGroup {
    /// Initialize process group.
    pub fn init(config: DistributedConfig) -> Result<Self> {
        if config.rank >= config.world_size {
            return Err(Error::InvalidConfig(format!(
                "Rank {} >= world_size {}",
                config.rank, config.world_size
            )));
        }

        Ok(Self {
            config,
            initialized: true,
            barrier_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get current rank.
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Check if this is main process.
    pub fn is_main_process(&self) -> bool {
        self.config.is_main_process()
    }

    /// Barrier synchronization.
    pub fn barrier(&self) -> Result<()> {
        if !self.initialized {
            return Err(Error::InvalidConfig("Process group not initialized".into()));
        }

        // In simulated mode, barrier is a no-op for single process
        if self.config.backend == DistributedBackend::Simulated {
            return Ok(());
        }

        // Real implementation would wait for all processes
        let mut counter = self.barrier_counter.lock().unwrap();
        *counter += 1;
        Ok(())
    }

    /// Destroy process group.
    pub fn destroy(&mut self) {
        self.initialized = false;
    }
}

/// All-reduce operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Mean reduction.
    Mean,
    /// Max reduction.
    Max,
    /// Min reduction.
    Min,
}

/// Gradient bucket for efficient communication.
#[derive(Debug)]
pub struct GradientBucket {
    /// Bucket data.
    data: Vec<f32>,
    /// Capacity in elements.
    capacity: usize,
    /// Current size.
    size: usize,
    /// Whether bucket is ready for communication.
    ready: bool,
}

impl GradientBucket {
    /// Create new bucket with given capacity in MB.
    pub fn new(capacity_mb: usize) -> Self {
        let capacity = capacity_mb * 1024 * 1024 / std::mem::size_of::<f32>();
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            size: 0,
            ready: false,
        }
    }

    /// Add gradients to bucket.
    pub fn add(&mut self, gradients: &[f32]) -> bool {
        if self.size + gradients.len() > self.capacity {
            return false;
        }
        self.data.extend_from_slice(gradients);
        self.size += gradients.len();
        true
    }

    /// Check if bucket is full.
    pub fn is_full(&self) -> bool {
        self.size >= self.capacity
    }

    /// Get bucket data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Clear bucket.
    pub fn clear(&mut self) {
        self.data.clear();
        self.size = 0;
        self.ready = false;
    }
}

/// Distributed Data Parallel (DDP) wrapper.
pub struct DistributedDataParallel {
    /// Process group.
    process_group: ProcessGroup,
    /// Gradient buckets.
    buckets: Vec<GradientBucket>,
    /// Statistics.
    stats: DDPStats,
    /// Configuration.
    config: DDPConfig,
}

/// DDP configuration.
#[derive(Debug, Clone)]
pub struct DDPConfig {
    /// Number of gradient buckets.
    pub num_buckets: usize,
    /// Bucket size in MB.
    pub bucket_size_mb: usize,
    /// Use gradient compression.
    pub gradient_compression: bool,
    /// Compression threshold (sparsity).
    pub compression_threshold: f32,
    /// Overlap communication with computation.
    pub overlap_comm: bool,
    /// Find unused parameters.
    pub find_unused_parameters: bool,
    /// Broadcast buffers.
    pub broadcast_buffers: bool,
}

impl Default for DDPConfig {
    fn default() -> Self {
        Self {
            num_buckets: 4,
            bucket_size_mb: 25,
            gradient_compression: false,
            compression_threshold: 0.001,
            overlap_comm: true,
            find_unused_parameters: false,
            broadcast_buffers: true,
        }
    }
}

/// DDP statistics.
#[derive(Debug, Clone, Default)]
pub struct DDPStats {
    /// Number of all-reduce calls.
    pub all_reduce_calls: usize,
    /// Total bytes communicated.
    pub bytes_communicated: usize,
    /// Total communication time.
    pub comm_time: Duration,
    /// Average gradient norm before sync.
    pub avg_grad_norm: f32,
}

impl DistributedDataParallel {
    /// Create new DDP wrapper.
    pub fn new(process_group: ProcessGroup, config: DDPConfig) -> Self {
        let buckets = (0..config.num_buckets)
            .map(|_| GradientBucket::new(config.bucket_size_mb))
            .collect();

        Self {
            process_group,
            buckets,
            stats: DDPStats::default(),
            config,
        }
    }

    /// Synchronize gradients across all processes.
    pub fn sync_gradients(&mut self, gradients: &mut [f32]) -> Result<()> {
        let start = Instant::now();

        // Calculate gradient norm
        let grad_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        self.stats.avg_grad_norm =
            (self.stats.avg_grad_norm * self.stats.all_reduce_calls as f32 + grad_norm)
            / (self.stats.all_reduce_calls + 1) as f32;

        // Apply compression if enabled
        if self.config.gradient_compression {
            self.compress_gradients(gradients);
        }

        // Perform all-reduce
        self.all_reduce(gradients, ReduceOp::Mean)?;

        // Update stats
        self.stats.all_reduce_calls += 1;
        self.stats.bytes_communicated += gradients.len() * std::mem::size_of::<f32>();
        self.stats.comm_time += start.elapsed();

        Ok(())
    }

    /// All-reduce operation.
    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        let world_size = self.process_group.world_size();

        if world_size == 1 {
            return Ok(());
        }

        // In simulated mode, just simulate the reduction
        match op {
            ReduceOp::Mean => {
                for val in data.iter_mut() {
                    *val /= world_size as f32;
                }
            }
            ReduceOp::Sum => {
                // Already summed in a real scenario
            }
            ReduceOp::Max | ReduceOp::Min => {
                // Already reduced
            }
        }

        Ok(())
    }

    /// Compress gradients using top-k sparsification.
    fn compress_gradients(&self, gradients: &mut [f32]) {
        let threshold = self.config.compression_threshold;
        for g in gradients.iter_mut() {
            if g.abs() < threshold {
                *g = 0.0;
            }
        }
    }

    /// Broadcast parameters from rank 0 to all other ranks.
    pub fn broadcast_parameters(&mut self, params: &mut [f32]) -> Result<()> {
        if self.process_group.world_size() == 1 {
            return Ok(());
        }

        // In a real implementation, rank 0 would broadcast to all others
        // For simulation, parameters are already synchronized
        Ok(())
    }

    /// Get statistics.
    pub fn stats(&self) -> &DDPStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DDPStats::default();
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.process_group.world_size()
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        self.process_group.rank()
    }
}

/// Fully Sharded Data Parallel (FSDP) configuration.
#[derive(Debug, Clone)]
pub struct FSDPConfig {
    /// Sharding strategy.
    pub sharding_strategy: ShardingStrategy,
    /// Auto wrap policy.
    pub auto_wrap_policy: AutoWrapPolicy,
    /// Mixed precision policy.
    pub mixed_precision: Option<MixedPrecisionPolicy>,
    /// CPU offload.
    pub cpu_offload: bool,
    /// Backward prefetch.
    pub backward_prefetch: BackwardPrefetch,
    /// Activation checkpointing.
    pub activation_checkpointing: bool,
}

impl Default for FSDPConfig {
    fn default() -> Self {
        Self {
            sharding_strategy: ShardingStrategy::FullShard,
            auto_wrap_policy: AutoWrapPolicy::SizeBasedWrap { min_params: 100_000 },
            mixed_precision: None,
            cpu_offload: false,
            backward_prefetch: BackwardPrefetch::BackwardPre,
            activation_checkpointing: false,
        }
    }
}

/// Sharding strategy for FSDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Shard parameters, gradients, and optimizer states.
    FullShard,
    /// Shard gradients and optimizer states.
    ShardGradOp,
    /// No sharding (like DDP).
    NoShard,
    /// Hybrid sharding across nodes.
    HybridShard,
}

/// Auto wrap policy.
#[derive(Debug, Clone)]
pub enum AutoWrapPolicy {
    /// No automatic wrapping.
    Never,
    /// Wrap layers with at least min_params parameters.
    SizeBasedWrap { min_params: usize },
    /// Wrap specific transformer layers.
    TransformerWrap { layer_names: Vec<String> },
}

/// Mixed precision policy.
#[derive(Debug, Clone)]
pub struct MixedPrecisionPolicy {
    /// Parameter dtype.
    pub param_dtype: DataType,
    /// Buffer dtype.
    pub buffer_dtype: DataType,
    /// Reduce dtype.
    pub reduce_dtype: DataType,
}

/// Data type for mixed precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
}

/// Backward prefetch strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardPrefetch {
    /// Prefetch before backward.
    BackwardPre,
    /// Prefetch after backward.
    BackwardPost,
    /// No prefetching.
    None,
}

/// FSDP wrapper for fully sharded data parallelism.
pub struct FullyShardedDataParallel {
    /// Process group.
    process_group: ProcessGroup,
    /// Configuration.
    config: FSDPConfig,
    /// Sharded parameters.
    sharded_params: Vec<ShardedParameter>,
    /// Statistics.
    stats: FSDPStats,
}

/// A sharded parameter.
#[derive(Debug)]
pub struct ShardedParameter {
    /// Parameter name.
    pub name: String,
    /// Local shard of the parameter.
    pub local_shard: Vec<f32>,
    /// Full parameter shape.
    pub full_shape: Vec<usize>,
    /// Shard offsets.
    pub shard_offset: usize,
    /// Shard size.
    pub shard_size: usize,
}

/// FSDP statistics.
#[derive(Debug, Clone, Default)]
pub struct FSDPStats {
    /// Number of all-gather calls.
    pub all_gather_calls: usize,
    /// Number of reduce-scatter calls.
    pub reduce_scatter_calls: usize,
    /// Total bytes communicated.
    pub bytes_communicated: usize,
    /// Peak memory usage.
    pub peak_memory_bytes: usize,
}

impl FullyShardedDataParallel {
    /// Create new FSDP wrapper.
    pub fn new(process_group: ProcessGroup, config: FSDPConfig) -> Self {
        Self {
            process_group,
            config,
            sharded_params: Vec::new(),
            stats: FSDPStats::default(),
        }
    }

    /// Shard parameters across processes.
    pub fn shard_parameters(&mut self, params: &[(&str, Vec<f32>)]) -> Result<()> {
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();

        self.sharded_params.clear();

        for (name, param) in params {
            let full_size = param.len();
            let shard_size = (full_size + world_size - 1) / world_size;
            let shard_offset = rank * shard_size;
            let actual_size = (shard_size).min(full_size.saturating_sub(shard_offset));

            let local_shard = if shard_offset < full_size {
                param[shard_offset..shard_offset + actual_size].to_vec()
            } else {
                Vec::new()
            };

            self.sharded_params.push(ShardedParameter {
                name: name.to_string(),
                local_shard,
                full_shape: vec![full_size],
                shard_offset,
                shard_size: actual_size,
            });
        }

        Ok(())
    }

    /// All-gather to reconstruct full parameters.
    pub fn all_gather(&mut self, param_index: usize) -> Result<Vec<f32>> {
        if param_index >= self.sharded_params.len() {
            return Err(Error::InvalidConfig("Invalid parameter index".into()));
        }

        let world_size = self.process_group.world_size();
        let param = &self.sharded_params[param_index];

        // Simulate all-gather
        let mut full_param = vec![0.0; param.full_shape.iter().product()];

        // Copy local shard to full parameter
        if param.shard_offset < full_param.len() {
            let end = (param.shard_offset + param.local_shard.len()).min(full_param.len());
            full_param[param.shard_offset..end]
                .copy_from_slice(&param.local_shard[..end - param.shard_offset]);
        }

        self.stats.all_gather_calls += 1;
        self.stats.bytes_communicated += full_param.len() * std::mem::size_of::<f32>();

        Ok(full_param)
    }

    /// Reduce-scatter gradients.
    pub fn reduce_scatter(&mut self, gradients: Vec<f32>, param_index: usize) -> Result<Vec<f32>> {
        if param_index >= self.sharded_params.len() {
            return Err(Error::InvalidConfig("Invalid parameter index".into()));
        }

        let world_size = self.process_group.world_size();
        let param = &self.sharded_params[param_index];

        // Simulate reduce-scatter: each rank gets its shard of reduced gradients
        let shard_size = param.shard_size;
        let offset = param.shard_offset;

        let reduced_shard: Vec<f32> = if offset < gradients.len() {
            let end = (offset + shard_size).min(gradients.len());
            gradients[offset..end]
                .iter()
                .map(|g| g / world_size as f32)
                .collect()
        } else {
            vec![0.0; shard_size]
        };

        self.stats.reduce_scatter_calls += 1;
        self.stats.bytes_communicated += gradients.len() * std::mem::size_of::<f32>();

        Ok(reduced_shard)
    }

    /// Get statistics.
    pub fn stats(&self) -> &FSDPStats {
        &self.stats
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.process_group.world_size()
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        self.process_group.rank()
    }
}

/// Distributed sampler for data loading.
pub struct DistributedSampler {
    /// Total dataset size.
    dataset_size: usize,
    /// Number of replicas (world size).
    num_replicas: usize,
    /// Current rank.
    rank: usize,
    /// Shuffle data.
    shuffle: bool,
    /// Random seed.
    seed: u64,
    /// Current epoch.
    epoch: usize,
    /// Drop last incomplete batch.
    drop_last: bool,
}

impl DistributedSampler {
    /// Create new distributed sampler.
    pub fn new(
        dataset_size: usize,
        num_replicas: usize,
        rank: usize,
        shuffle: bool,
        seed: u64,
    ) -> Self {
        Self {
            dataset_size,
            num_replicas,
            rank,
            shuffle,
            seed,
            epoch: 0,
            drop_last: false,
        }
    }

    /// Set epoch for shuffling.
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    /// Get number of samples for this rank.
    pub fn num_samples(&self) -> usize {
        let total_size = if self.drop_last {
            self.dataset_size - (self.dataset_size % self.num_replicas)
        } else {
            self.dataset_size + (self.num_replicas - self.dataset_size % self.num_replicas) % self.num_replicas
        };
        total_size / self.num_replicas
    }

    /// Get indices for this rank.
    pub fn indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.dataset_size).collect();

        if self.shuffle {
            // Simple deterministic shuffle based on seed + epoch
            let seed = self.seed.wrapping_add(self.epoch as u64);
            let mut rng_state = seed;
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        // Pad if necessary
        let total_size = self.num_samples() * self.num_replicas;
        while indices.len() < total_size {
            let idx = indices.len() % self.dataset_size;
            indices.push(indices[idx]);
        }

        // Subsample for this rank
        indices
            .into_iter()
            .skip(self.rank)
            .step_by(self.num_replicas)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert!(config.is_main_process());
    }

    #[test]
    fn test_process_group_init() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        assert_eq!(pg.world_size(), 1);
        assert_eq!(pg.rank(), 0);
    }

    #[test]
    fn test_process_group_barrier() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        assert!(pg.barrier().is_ok());
    }

    #[test]
    fn test_gradient_bucket() {
        let mut bucket = GradientBucket::new(1); // 1MB
        let grads = vec![1.0; 1000];
        assert!(bucket.add(&grads));
        assert!(!bucket.is_full());
        assert_eq!(bucket.data().len(), 1000);
        bucket.clear();
        assert_eq!(bucket.data().len(), 0);
    }

    #[test]
    fn test_ddp_creation() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let ddp_config = DDPConfig::default();
        let ddp = DistributedDataParallel::new(pg, ddp_config);
        assert_eq!(ddp.world_size(), 1);
    }

    #[test]
    fn test_ddp_sync_gradients() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let ddp_config = DDPConfig::default();
        let mut ddp = DistributedDataParallel::new(pg, ddp_config);

        let mut grads = vec![1.0, 2.0, 3.0, 4.0];
        assert!(ddp.sync_gradients(&mut grads).is_ok());
        assert_eq!(ddp.stats().all_reduce_calls, 1);
    }

    #[test]
    fn test_ddp_with_compression() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let mut ddp_config = DDPConfig::default();
        ddp_config.gradient_compression = true;
        ddp_config.compression_threshold = 0.5;

        let mut ddp = DistributedDataParallel::new(pg, ddp_config);

        let mut grads = vec![0.1, 0.2, 1.0, 2.0];
        assert!(ddp.sync_gradients(&mut grads).is_ok());

        // Small gradients should be zeroed
        assert_eq!(grads[0], 0.0);
        assert_eq!(grads[1], 0.0);
    }

    #[test]
    fn test_fsdp_creation() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let fsdp_config = FSDPConfig::default();
        let fsdp = FullyShardedDataParallel::new(pg, fsdp_config);
        assert_eq!(fsdp.world_size(), 1);
    }

    #[test]
    fn test_fsdp_shard_parameters() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let fsdp_config = FSDPConfig::default();
        let mut fsdp = FullyShardedDataParallel::new(pg, fsdp_config);

        let params = vec![
            ("layer1.weight", vec![1.0; 100]),
            ("layer1.bias", vec![0.0; 10]),
        ];
        assert!(fsdp.shard_parameters(&params).is_ok());
    }

    #[test]
    fn test_fsdp_all_gather() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let fsdp_config = FSDPConfig::default();
        let mut fsdp = FullyShardedDataParallel::new(pg, fsdp_config);

        let params = vec![("weight", vec![1.0, 2.0, 3.0, 4.0])];
        fsdp.shard_parameters(&params).unwrap();

        let full = fsdp.all_gather(0).unwrap();
        assert_eq!(full.len(), 4);
    }

    #[test]
    fn test_distributed_sampler() {
        let sampler = DistributedSampler::new(100, 4, 0, false, 42);
        assert_eq!(sampler.num_samples(), 25);

        let indices = sampler.indices();
        assert_eq!(indices.len(), 25);

        // Check that indices are correct for rank 0
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 4);
    }

    #[test]
    fn test_distributed_sampler_shuffle() {
        let mut sampler = DistributedSampler::new(100, 2, 0, true, 42);

        let indices_epoch_0 = sampler.indices();
        sampler.set_epoch(1);
        let indices_epoch_1 = sampler.indices();

        // Indices should differ between epochs
        assert_ne!(indices_epoch_0, indices_epoch_1);
    }

    #[test]
    fn test_multi_rank_simulation() {
        // Simulate 4 ranks
        let world_size = 4;

        for rank in 0..world_size {
            let config = DistributedConfig {
                world_size,
                rank,
                ..Default::default()
            };
            let pg = ProcessGroup::init(config).unwrap();
            let ddp = DistributedDataParallel::new(pg, DDPConfig::default());

            assert_eq!(ddp.world_size(), 4);
            assert_eq!(ddp.rank(), rank);
        }
    }

    #[test]
    fn test_sharding_strategy() {
        let full = ShardingStrategy::FullShard;
        let hybrid = ShardingStrategy::HybridShard;
        assert_ne!(full, hybrid);
    }

    #[test]
    fn test_ddp_stats() {
        let config = DistributedConfig::default();
        let pg = ProcessGroup::init(config).unwrap();
        let mut ddp = DistributedDataParallel::new(pg, DDPConfig::default());

        let mut grads = vec![1.0; 1000];
        for _ in 0..5 {
            ddp.sync_gradients(&mut grads).unwrap();
        }

        let stats = ddp.stats();
        assert_eq!(stats.all_reduce_calls, 5);
        assert!(stats.bytes_communicated > 0);
    }

    #[test]
    fn test_process_group_invalid_rank() {
        let config = DistributedConfig {
            world_size: 4,
            rank: 5, // Invalid
            ..Default::default()
        };
        assert!(ProcessGroup::init(config).is_err());
    }
}
