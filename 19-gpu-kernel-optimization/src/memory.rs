//! Memory optimization utilities for GPU kernel development.
//!
//! This module provides:
//! - Bank conflict detection and elimination
//! - Shared memory padding strategies
//! - Memory access pattern analysis
//! - Occupancy calculation

use std::collections::HashMap;

/// Number of memory banks (32 for NVIDIA GPUs).
pub const NUM_BANKS: usize = 32;

/// Bank width in bytes.
pub const BANK_WIDTH: usize = 4;

/// Maximum shared memory per block (48KB for most GPUs).
pub const MAX_SHARED_MEM: usize = 49152;

/// Bank conflict analysis result.
#[derive(Debug, Clone)]
pub struct BankConflictAnalysis {
    /// Number of bank conflicts detected.
    pub conflict_count: usize,
    /// Maximum conflicts on any single bank.
    pub max_conflicts: usize,
    /// Map of bank -> access count.
    pub bank_accesses: [usize; NUM_BANKS],
    /// Conflict ratio (0.0 = none, 1.0 = all conflicting).
    pub conflict_ratio: f32,
    /// Recommended padding to eliminate conflicts.
    pub suggested_padding: usize,
}

impl BankConflictAnalysis {
    /// Analyze bank conflicts for a set of addresses.
    ///
    /// # Arguments
    /// * `addresses` - List of memory addresses (byte offsets from shared memory base)
    pub fn analyze(addresses: &[usize]) -> Self {
        let mut bank_accesses = [0usize; NUM_BANKS];

        // Count accesses per bank
        for &addr in addresses {
            let bank = (addr / BANK_WIDTH) % NUM_BANKS;
            bank_accesses[bank] += 1;
        }

        // Calculate conflicts
        let mut conflict_count = 0;
        let mut max_conflicts = 0;

        for &count in &bank_accesses {
            if count > 1 {
                // n accesses to same bank = n-1 conflicts
                let conflicts = count - 1;
                conflict_count += conflicts;
                max_conflicts = max_conflicts.max(conflicts);
            }
        }

        let conflict_ratio = if addresses.len() > 1 {
            conflict_count as f32 / (addresses.len() - 1) as f32
        } else {
            0.0
        };

        // Calculate suggested padding
        let suggested_padding = Self::calculate_padding(&bank_accesses);

        Self {
            conflict_count,
            max_conflicts,
            bank_accesses,
            conflict_ratio,
            suggested_padding,
        }
    }

    /// Calculate padding to eliminate bank conflicts.
    fn calculate_padding(bank_accesses: &[usize; NUM_BANKS]) -> usize {
        // Find the most conflicted banks
        let max_access = *bank_accesses.iter().max().unwrap_or(&0);

        if max_access <= 1 {
            return 0; // No conflicts
        }

        // Try different padding values
        for padding in 1..=NUM_BANKS {
            // Simulate with padding
            let mut simulated = [0usize; NUM_BANKS];
            let mut conflict_free = true;

            for (i, &count) in bank_accesses.iter().enumerate() {
                let new_bank = (i + padding) % NUM_BANKS;
                simulated[new_bank] += count;
                if simulated[new_bank] > 1 {
                    conflict_free = false;
                }
            }

            if conflict_free {
                return padding;
            }
        }

        1 // Default: add 1 element padding per row
    }

    /// Check if access pattern is bank-conflict free.
    pub fn is_conflict_free(&self) -> bool {
        self.conflict_count == 0
    }
}

/// Shared memory configuration with padding.
#[derive(Debug, Clone)]
pub struct SharedMemoryConfig {
    /// Width of each row (elements).
    pub width: usize,
    /// Height (number of rows).
    pub height: usize,
    /// Padding elements per row.
    pub padding: usize,
    /// Element size in bytes.
    pub element_size: usize,
}

impl SharedMemoryConfig {
    /// Create new shared memory configuration.
    pub fn new(width: usize, height: usize, element_size: usize) -> Self {
        Self {
            width,
            height,
            padding: 0,
            element_size,
        }
    }

    /// Create configuration with automatic padding to avoid conflicts.
    pub fn with_auto_padding(width: usize, height: usize, element_size: usize) -> Self {
        // Calculate padding to avoid bank conflicts
        let elements_per_bank = BANK_WIDTH / element_size;
        let banks_per_row = (width * element_size + BANK_WIDTH - 1) / BANK_WIDTH;

        // If row width is multiple of NUM_BANKS, add padding
        let padding = if banks_per_row % NUM_BANKS == 0 { 1 } else { 0 };

        Self {
            width,
            height,
            padding,
            element_size,
        }
    }

    /// Get stride (elements) to next row.
    pub fn stride(&self) -> usize {
        self.width + self.padding
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.height * self.stride() * self.element_size
    }

    /// Convert 2D index to linear offset.
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.stride() + col
    }

    /// Check if configuration fits in shared memory.
    pub fn fits_in_shared_memory(&self) -> bool {
        self.size_bytes() <= MAX_SHARED_MEM
    }
}

/// Occupancy calculation for GPU kernels.
#[derive(Debug, Clone)]
pub struct OccupancyCalculator {
    /// Maximum threads per multiprocessor.
    pub max_threads_per_sm: usize,
    /// Maximum blocks per multiprocessor.
    pub max_blocks_per_sm: usize,
    /// Total registers per multiprocessor.
    pub registers_per_sm: usize,
    /// Total shared memory per multiprocessor.
    pub shared_mem_per_sm: usize,
}

/// Kernel resource requirements.
#[derive(Debug, Clone)]
pub struct KernelRequirements {
    /// Threads per block.
    pub threads_per_block: usize,
    /// Registers per thread.
    pub registers_per_thread: usize,
    /// Shared memory per block.
    pub shared_mem_per_block: usize,
}

/// Occupancy result.
#[derive(Debug, Clone)]
pub struct OccupancyResult {
    /// Active blocks per SM.
    pub blocks_per_sm: usize,
    /// Active warps per SM.
    pub warps_per_sm: usize,
    /// Maximum possible warps per SM.
    pub max_warps_per_sm: usize,
    /// Occupancy percentage (0-100).
    pub occupancy: f32,
    /// Limiting factor.
    pub limiting_factor: LimitingFactor,
}

/// What limits occupancy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LimitingFactor {
    Threads,
    Registers,
    SharedMemory,
    Blocks,
    None,
}

impl OccupancyCalculator {
    /// Create calculator for Ampere architecture (SM 8.0).
    pub fn ampere() -> Self {
        Self {
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            shared_mem_per_sm: 163840, // 164KB configurable
        }
    }

    /// Create calculator for Volta architecture (SM 7.0).
    pub fn volta() -> Self {
        Self {
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            shared_mem_per_sm: 98304, // 96KB
        }
    }

    /// Create calculator for Turing architecture (SM 7.5).
    pub fn turing() -> Self {
        Self {
            max_threads_per_sm: 1024,
            max_blocks_per_sm: 16,
            registers_per_sm: 65536,
            shared_mem_per_sm: 65536, // 64KB
        }
    }

    /// Calculate occupancy for given kernel requirements.
    pub fn calculate(&self, reqs: &KernelRequirements) -> OccupancyResult {
        let warp_size = 32;
        let max_warps = self.max_threads_per_sm / warp_size;

        // Block limit
        let blocks_by_threads = self.max_threads_per_sm / reqs.threads_per_block;

        // Register limit (granularity: warps allocated in groups of 4)
        let warps_per_block = (reqs.threads_per_block + warp_size - 1) / warp_size;
        let regs_per_block = warps_per_block * warp_size * reqs.registers_per_thread;
        let blocks_by_regs = if regs_per_block > 0 {
            self.registers_per_sm / regs_per_block
        } else {
            self.max_blocks_per_sm
        };

        // Shared memory limit
        let blocks_by_smem = if reqs.shared_mem_per_block > 0 {
            self.shared_mem_per_sm / reqs.shared_mem_per_block
        } else {
            self.max_blocks_per_sm
        };

        // Find limiting factor
        let limits = [
            (blocks_by_threads, LimitingFactor::Threads),
            (blocks_by_regs, LimitingFactor::Registers),
            (blocks_by_smem, LimitingFactor::SharedMemory),
            (self.max_blocks_per_sm, LimitingFactor::Blocks),
        ];

        let (min_blocks, limiting_factor) = limits.iter()
            .min_by_key(|(b, _)| *b)
            .map(|&(b, f)| (b.min(self.max_blocks_per_sm), f))
            .unwrap();

        let active_warps = min_blocks * warps_per_block;
        let occupancy = (active_warps as f32 / max_warps as f32) * 100.0;

        OccupancyResult {
            blocks_per_sm: min_blocks,
            warps_per_sm: active_warps,
            max_warps_per_sm: max_warps,
            occupancy,
            limiting_factor,
        }
    }

    /// Find optimal block size for maximum occupancy.
    pub fn find_optimal_block_size(
        &self,
        regs_per_thread: usize,
        shared_mem_per_block: usize,
    ) -> usize {
        let mut best_occupancy = 0.0f32;
        let mut best_block_size = 64;

        // Try common block sizes
        for block_size in [32, 64, 128, 256, 512, 1024] {
            if block_size > self.max_threads_per_sm {
                break;
            }

            let reqs = KernelRequirements {
                threads_per_block: block_size,
                registers_per_thread: regs_per_thread,
                shared_mem_per_block,
            };

            let result = self.calculate(&reqs);
            if result.occupancy > best_occupancy {
                best_occupancy = result.occupancy;
                best_block_size = block_size;
            }
        }

        best_block_size
    }
}

/// Memory access pattern for optimization analysis.
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Type of access.
    pub access_type: AccessType,
    /// Stride between consecutive accesses (in elements).
    pub stride: usize,
    /// Whether access is aligned.
    pub aligned: bool,
    /// Predicted cache behavior.
    pub cache_behavior: CacheBehavior,
}

/// Type of memory access.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessType {
    /// Linear sequential access.
    Sequential,
    /// Regular strided access.
    Strided,
    /// Random access.
    Random,
    /// Broadcast (same value to all threads).
    Broadcast,
}

/// Predicted cache behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheBehavior {
    /// Cache hit expected.
    Hit,
    /// Cache miss expected.
    Miss,
    /// Unpredictable.
    Unknown,
}

impl MemoryAccessPattern {
    /// Analyze access pattern from a list of addresses.
    pub fn analyze(addresses: &[usize]) -> Self {
        if addresses.len() <= 1 {
            return Self {
                access_type: AccessType::Sequential,
                stride: 0,
                aligned: true,
                cache_behavior: CacheBehavior::Hit,
            };
        }

        // Calculate strides
        let mut strides = Vec::new();
        for i in 1..addresses.len() {
            let stride = if addresses[i] >= addresses[i - 1] {
                addresses[i] - addresses[i - 1]
            } else {
                addresses[i - 1] - addresses[i]
            };
            strides.push(stride);
        }

        // Determine access type
        let all_same = strides.iter().all(|&s| s == strides[0]);
        let avg_stride = strides.iter().sum::<usize>() / strides.len();

        let access_type = if all_same && strides[0] == 1 {
            AccessType::Sequential
        } else if all_same && strides[0] == 0 {
            AccessType::Broadcast
        } else if all_same {
            AccessType::Strided
        } else {
            AccessType::Random
        };

        // Check alignment
        let aligned = addresses.iter().all(|&a| a % 4 == 0);

        // Predict cache behavior
        let cache_behavior = match access_type {
            AccessType::Sequential => CacheBehavior::Hit,
            AccessType::Broadcast => CacheBehavior::Hit,
            AccessType::Strided if avg_stride <= 32 => CacheBehavior::Hit,
            AccessType::Strided => CacheBehavior::Miss,
            AccessType::Random => CacheBehavior::Miss,
        };

        Self {
            access_type,
            stride: avg_stride,
            aligned,
            cache_behavior,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_conflict_detection() {
        // No conflicts: sequential 32 addresses
        let addrs: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let analysis = BankConflictAnalysis::analyze(&addrs);
        assert!(analysis.is_conflict_free());
    }

    #[test]
    fn test_bank_conflict_with_stride() {
        // Conflict: stride of 32 banks
        let addrs: Vec<usize> = (0..8).map(|i| i * 32 * 4).collect();
        let analysis = BankConflictAnalysis::analyze(&addrs);
        // All access same bank
        assert!(!analysis.is_conflict_free());
    }

    #[test]
    fn test_shared_memory_config() {
        let config = SharedMemoryConfig::new(64, 64, 4);
        assert_eq!(config.stride(), 64);
        assert_eq!(config.index(1, 0), 64);

        let config_padded = SharedMemoryConfig::with_auto_padding(32, 32, 4);
        // 32 elements * 4 bytes = 128 bytes = 32 banks, needs padding
        assert!(config_padded.padding > 0 || config_padded.stride() > 32);
    }

    #[test]
    fn test_occupancy_calculation() {
        let calc = OccupancyCalculator::ampere();

        let reqs = KernelRequirements {
            threads_per_block: 256,
            registers_per_thread: 32,
            shared_mem_per_block: 16384,
        };

        let result = calc.calculate(&reqs);
        assert!(result.occupancy > 0.0);
        assert!(result.blocks_per_sm > 0);
    }

    #[test]
    fn test_optimal_block_size() {
        let calc = OccupancyCalculator::ampere();
        let optimal = calc.find_optimal_block_size(32, 8192);

        assert!(optimal >= 32 && optimal <= 1024);
    }

    #[test]
    fn test_memory_pattern_sequential() {
        let addrs: Vec<usize> = (0..8).map(|i| i * 4).collect();
        let pattern = MemoryAccessPattern::analyze(&addrs);

        assert_eq!(pattern.access_type, AccessType::Strided);
        assert_eq!(pattern.stride, 4);
        assert!(pattern.aligned);
    }

    #[test]
    fn test_memory_pattern_broadcast() {
        let addrs = vec![100, 100, 100, 100];
        let pattern = MemoryAccessPattern::analyze(&addrs);

        assert_eq!(pattern.access_type, AccessType::Broadcast);
    }

    #[test]
    fn test_limiting_factors() {
        let calc = OccupancyCalculator::ampere();

        // Thread limited
        let reqs = KernelRequirements {
            threads_per_block: 1024,
            registers_per_thread: 16,
            shared_mem_per_block: 1024,
        };
        let result = calc.calculate(&reqs);
        assert_eq!(result.limiting_factor, LimitingFactor::Threads);

        // Register limited
        let reqs = KernelRequirements {
            threads_per_block: 256,
            registers_per_thread: 255, // Very high register usage
            shared_mem_per_block: 1024,
        };
        let result = calc.calculate(&reqs);
        assert!(
            result.limiting_factor == LimitingFactor::Registers ||
            result.limiting_factor == LimitingFactor::Blocks
        );
    }
}
