//! NUMA-aware memory management and scheduling.
//!
//! This module provides abstractions for NUMA-aware operations,
//! including topology detection, local memory allocation, and
//! partition-based execution.
//!
//! Note: NUMA functionality is simulated on systems without libnuma.

use crate::{Error, Result, BLOCK_SIZE, CACHE_LINE_SIZE};
use std::sync::atomic::{AtomicUsize, Ordering};

/// NUMA node information.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID.
    pub id: usize,
    /// CPU cores on this node.
    pub cpus: Vec<usize>,
    /// Total memory on this node in bytes.
    pub total_memory: usize,
    /// Free memory on this node in bytes.
    pub free_memory: usize,
}

impl NumaNode {
    /// Create a new NUMA node.
    pub fn new(id: usize, cpus: Vec<usize>, total_memory: usize) -> Self {
        Self {
            id,
            cpus,
            total_memory,
            free_memory: total_memory,
        }
    }

    /// Get number of CPUs on this node.
    pub fn num_cpus(&self) -> usize {
        self.cpus.len()
    }
}

/// NUMA topology information.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes.
    pub num_nodes: usize,
    /// Information about each node.
    pub nodes: Vec<NumaNode>,
    /// Total number of CPUs.
    pub total_cpus: usize,
    /// Distance matrix between nodes (latency).
    pub distances: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Detect NUMA topology from system.
    pub fn detect() -> Self {
        // On macOS and systems without libnuma, simulate single-node topology
        Self::single_node()
    }

    /// Create single-node topology (for non-NUMA systems).
    pub fn single_node() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let node = NumaNode::new(
            0,
            (0..num_cpus).collect(),
            16 * 1024 * 1024 * 1024, // 16GB default
        );

        Self {
            num_nodes: 1,
            nodes: vec![node],
            total_cpus: num_cpus,
            distances: vec![vec![10]], // Local access latency
        }
    }

    /// Create multi-node topology for testing.
    pub fn simulated(num_nodes: usize, cpus_per_node: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        let mut cpu_id = 0;

        for node_id in 0..num_nodes {
            let cpus: Vec<usize> = (cpu_id..cpu_id + cpus_per_node).collect();
            cpu_id += cpus_per_node;
            nodes.push(NumaNode::new(node_id, cpus, 8 * 1024 * 1024 * 1024));
        }

        // Create distance matrix
        let mut distances = vec![vec![0u32; num_nodes]; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    distances[i][j] = 10; // Local access
                } else {
                    distances[i][j] = 20; // Remote access
                }
            }
        }

        Self {
            num_nodes,
            nodes,
            total_cpus: num_nodes * cpus_per_node,
            distances,
        }
    }

    /// Get the node for a given CPU.
    pub fn cpu_to_node(&self, cpu: usize) -> Option<usize> {
        for node in &self.nodes {
            if node.cpus.contains(&cpu) {
                return Some(node.id);
            }
        }
        None
    }

    /// Get distance between two nodes.
    pub fn distance(&self, node1: usize, node2: usize) -> Option<u32> {
        if node1 < self.num_nodes && node2 < self.num_nodes {
            Some(self.distances[node1][node2])
        } else {
            None
        }
    }

    /// Check if running on single-node system.
    pub fn is_single_node(&self) -> bool {
        self.num_nodes == 1
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

/// NUMA-aware allocator.
#[derive(Debug)]
pub struct NumaAllocator {
    /// Current topology.
    topology: NumaTopology,
    /// Allocation statistics per node.
    allocations: Vec<AtomicUsize>,
}

impl NumaAllocator {
    /// Create a new NUMA allocator.
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        let allocations = (0..topology.num_nodes)
            .map(|_| AtomicUsize::new(0))
            .collect();

        Self {
            topology,
            allocations,
        }
    }

    /// Create with specific topology.
    pub fn with_topology(topology: NumaTopology) -> Self {
        let allocations = (0..topology.num_nodes)
            .map(|_| AtomicUsize::new(0))
            .collect();

        Self {
            topology,
            allocations,
        }
    }

    /// Get topology.
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Allocate memory on specific node (simulated).
    pub fn alloc_on_node<T: Default + Clone>(&self, count: usize, node: usize) -> Result<Vec<T>> {
        if node >= self.topology.num_nodes {
            return Err(Error::InvalidOperation(format!(
                "Invalid NUMA node {}",
                node
            )));
        }

        // Track allocation
        self.allocations[node].fetch_add(count * std::mem::size_of::<T>(), Ordering::Relaxed);

        // Note: Real NUMA allocation would use numa_alloc_onnode
        // Here we just allocate normally
        Ok(vec![T::default(); count])
    }

    /// Allocate interleaved across nodes (simulated).
    pub fn alloc_interleaved<T: Default + Clone>(&self, count: usize) -> Vec<T> {
        let per_node = count / self.topology.num_nodes;
        for i in 0..self.topology.num_nodes {
            self.allocations[i].fetch_add(per_node * std::mem::size_of::<T>(), Ordering::Relaxed);
        }

        vec![T::default(); count]
    }

    /// Get total allocations on a node.
    pub fn node_allocations(&self, node: usize) -> usize {
        if node < self.allocations.len() {
            self.allocations[node].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get best node for current thread (simulated).
    pub fn current_node(&self) -> usize {
        // In a real implementation, this would return the actual NUMA node
        0
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// NUMA-partitioned data.
#[derive(Debug)]
pub struct NumaPartitionedVec<T> {
    /// Partitions per NUMA node.
    partitions: Vec<Vec<T>>,
    /// Total element count.
    total_len: usize,
}

impl<T: Clone> NumaPartitionedVec<T> {
    /// Create from data, partitioning across nodes.
    pub fn from_vec(data: Vec<T>, num_nodes: usize) -> Self {
        if num_nodes == 0 {
            return Self {
                partitions: vec![],
                total_len: 0,
            };
        }

        let total_len = data.len();
        let chunk_size = (total_len + num_nodes - 1) / num_nodes;
        let partitions: Vec<Vec<T>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

        Self {
            partitions,
            total_len,
        }
    }

    /// Create empty with specified number of partitions.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            partitions: vec![Vec::new(); num_partitions],
            total_len: 0,
        }
    }

    /// Get number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get total length.
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Get partition.
    pub fn partition(&self, index: usize) -> Option<&[T]> {
        self.partitions.get(index).map(|v| v.as_slice())
    }

    /// Get mutable partition.
    pub fn partition_mut(&mut self, index: usize) -> Option<&mut Vec<T>> {
        self.partitions.get_mut(index)
    }

    /// Push to specific partition.
    pub fn push_to_partition(&mut self, partition: usize, value: T) -> Result<()> {
        if partition >= self.partitions.len() {
            return Err(Error::IndexOutOfBounds {
                index: partition,
                len: self.partitions.len(),
            });
        }
        self.partitions[partition].push(value);
        self.total_len += 1;
        Ok(())
    }

    /// Collect all partitions into a single vector.
    pub fn collect(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.total_len);
        for partition in &self.partitions {
            result.extend(partition.iter().cloned());
        }
        result
    }

    /// Iterate over all elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.partitions.iter().flat_map(|p| p.iter())
    }
}

/// NUMA-aware executor.
#[derive(Debug)]
pub struct NumaExecutor {
    /// NUMA allocator.
    allocator: NumaAllocator,
}

impl NumaExecutor {
    /// Create a new NUMA executor.
    pub fn new() -> Self {
        Self {
            allocator: NumaAllocator::new(),
        }
    }

    /// Create with specific topology.
    pub fn with_topology(topology: NumaTopology) -> Self {
        Self {
            allocator: NumaAllocator::with_topology(topology),
        }
    }

    /// Get topology.
    pub fn topology(&self) -> &NumaTopology {
        self.allocator.topology()
    }

    /// Execute function on each partition in parallel.
    pub fn parallel_partitions<T, R, F>(&self, data: &NumaPartitionedVec<T>, f: F) -> Vec<R>
    where
        T: Clone + Send + Sync,
        R: Send,
        F: Fn(&[T], usize) -> R + Send + Sync,
    {
        use rayon::prelude::*;

        data.partitions
            .par_iter()
            .enumerate()
            .map(|(node, partition)| f(partition, node))
            .collect()
    }

    /// Sum f64 values across partitions.
    pub fn parallel_sum(&self, data: &NumaPartitionedVec<f64>) -> f64 {
        let partial_sums: Vec<f64> = self.parallel_partitions(data, |partition, _| {
            partition.iter().sum()
        });
        partial_sums.iter().sum()
    }

    /// Aggregate f64 values across partitions.
    pub fn parallel_aggregate<F, R>(&self, data: &NumaPartitionedVec<f64>, f: F, combine: fn(R, R) -> R, identity: R) -> R
    where
        F: Fn(&[f64]) -> R + Send + Sync,
        R: Send + Copy,
    {
        let partial: Vec<R> = self.parallel_partitions(data, |partition, _| f(partition));
        partial.into_iter().fold(identity, combine)
    }

    /// Get allocator.
    pub fn allocator(&self) -> &NumaAllocator {
        &self.allocator
    }
}

impl Default for NumaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Affinity settings for thread pinning.
#[derive(Debug, Clone)]
pub struct AffinitySettings {
    /// CPU cores to use.
    pub cores: Vec<usize>,
    /// Whether to enable pinning.
    pub enable_pinning: bool,
}

impl AffinitySettings {
    /// Create settings for all cores.
    pub fn all_cores() -> Self {
        let topology = NumaTopology::detect();
        let cores: Vec<usize> = topology.nodes.iter().flat_map(|n| n.cpus.clone()).collect();
        Self {
            cores,
            enable_pinning: true,
        }
    }

    /// Create settings for specific NUMA node.
    pub fn for_node(node: usize) -> Self {
        let topology = NumaTopology::detect();
        let cores = topology
            .nodes
            .get(node)
            .map(|n| n.cpus.clone())
            .unwrap_or_default();
        Self {
            cores,
            enable_pinning: true,
        }
    }

    /// Create disabled affinity settings.
    pub fn disabled() -> Self {
        Self {
            cores: vec![],
            enable_pinning: false,
        }
    }

    /// Get number of cores.
    pub fn num_cores(&self) -> usize {
        self.cores.len()
    }
}

impl Default for AffinitySettings {
    fn default() -> Self {
        Self::all_cores()
    }
}

/// Memory bandwidth estimator.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// Estimated bandwidth per node in GB/s.
    bandwidth_per_node: Vec<f64>,
}

impl BandwidthEstimator {
    /// Create a new bandwidth estimator.
    pub fn new(topology: &NumaTopology) -> Self {
        // Default estimates for modern systems
        let bandwidth_per_node = vec![50.0; topology.num_nodes]; // 50 GB/s per node
        Self { bandwidth_per_node }
    }

    /// Estimate time to transfer bytes from one node to another.
    pub fn estimate_transfer_time(&self, bytes: usize, from_node: usize, to_node: usize, topology: &NumaTopology) -> f64 {
        let bandwidth = self.bandwidth_per_node.get(from_node).copied().unwrap_or(50.0);
        let distance_factor = if from_node == to_node {
            1.0
        } else {
            topology.distance(from_node, to_node).unwrap_or(20) as f64 / 10.0
        };

        let effective_bandwidth = bandwidth / distance_factor;
        (bytes as f64 / 1e9) / effective_bandwidth
    }

    /// Estimate local bandwidth.
    pub fn local_bandwidth(&self, node: usize) -> f64 {
        self.bandwidth_per_node.get(node).copied().unwrap_or(50.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_detect() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes >= 1);
        assert!(!topology.nodes.is_empty());
        assert!(topology.total_cpus >= 1);
    }

    #[test]
    fn test_numa_topology_simulated() {
        let topology = NumaTopology::simulated(4, 8);
        assert_eq!(topology.num_nodes, 4);
        assert_eq!(topology.total_cpus, 32);

        // Check distances
        assert_eq!(topology.distance(0, 0), Some(10));
        assert_eq!(topology.distance(0, 1), Some(20));
    }

    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAllocator::new();
        let data: Vec<f64> = allocator.alloc_on_node(1000, 0).unwrap();
        assert_eq!(data.len(), 1000);
        assert!(allocator.node_allocations(0) > 0);
    }

    #[test]
    fn test_numa_partitioned_vec() {
        let data: Vec<i32> = (0..100).collect();
        let partitioned = NumaPartitionedVec::from_vec(data, 4);

        assert_eq!(partitioned.num_partitions(), 4);
        assert_eq!(partitioned.len(), 100);

        let collected = partitioned.collect();
        assert_eq!(collected.len(), 100);
    }

    #[test]
    fn test_numa_executor_parallel_sum() {
        let executor = NumaExecutor::new();
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let partitioned = NumaPartitionedVec::from_vec(data, executor.topology().num_nodes);

        let sum = executor.parallel_sum(&partitioned);
        assert_eq!(sum, 5050.0); // Sum of 1..=100
    }

    #[test]
    fn test_affinity_settings() {
        let all_cores = AffinitySettings::all_cores();
        assert!(all_cores.num_cores() > 0);
        assert!(all_cores.enable_pinning);

        let disabled = AffinitySettings::disabled();
        assert!(!disabled.enable_pinning);
    }

    #[test]
    fn test_bandwidth_estimator() {
        let topology = NumaTopology::simulated(2, 4);
        let estimator = BandwidthEstimator::new(&topology);

        let local_time = estimator.estimate_transfer_time(1_000_000_000, 0, 0, &topology);
        let remote_time = estimator.estimate_transfer_time(1_000_000_000, 0, 1, &topology);

        // Remote should take longer
        assert!(remote_time > local_time);
    }
}
