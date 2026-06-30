//! Vectorized hash operations for analytics.
//!
//! Provides SIMD-optimized hashing, hash tables, and hash-based aggregation.

use crate::{Error, Result, BLOCK_SIZE, VECTOR_WIDTH};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// MurmurHash3 constants for vectorized hashing.
const MURMUR_C1: u64 = 0xff51afd7ed558ccd;
const MURMUR_C2: u64 = 0xc4ceb9fe1a85ec53;

/// Vectorized hash operations.
pub struct VectorizedHash;

impl VectorizedHash {
    /// Hash a single i64 value using MurmurHash3 finalizer.
    #[inline]
    pub fn hash_i64(value: i64) -> u64 {
        let mut h = value as u64;
        h ^= h >> 33;
        h = h.wrapping_mul(MURMUR_C1);
        h ^= h >> 33;
        h = h.wrapping_mul(MURMUR_C2);
        h ^= h >> 33;
        h
    }

    /// Hash a single u64 value.
    #[inline]
    pub fn hash_u64(value: u64) -> u64 {
        let mut h = value;
        h ^= h >> 33;
        h = h.wrapping_mul(MURMUR_C1);
        h ^= h >> 33;
        h = h.wrapping_mul(MURMUR_C2);
        h ^= h >> 33;
        h
    }

    /// Hash a single i32 value.
    #[inline]
    pub fn hash_i32(value: i32) -> u64 {
        Self::hash_i64(value as i64)
    }

    /// Vectorized hash of i64 array.
    pub fn hash_i64_vec(keys: &[i64]) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(keys.len());

        // Process in chunks for cache efficiency
        for chunk in keys.chunks(BLOCK_SIZE) {
            for &key in chunk {
                hashes.push(Self::hash_i64(key));
            }
        }

        hashes
    }

    /// Vectorized hash of i32 array.
    pub fn hash_i32_vec(keys: &[i32]) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(keys.len());

        for chunk in keys.chunks(BLOCK_SIZE) {
            for &key in chunk {
                hashes.push(Self::hash_i32(key));
            }
        }

        hashes
    }

    /// Hash bytes using FNV-1a.
    #[inline]
    pub fn hash_bytes(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Hash string slice.
    #[inline]
    pub fn hash_str(s: &str) -> u64 {
        Self::hash_bytes(s.as_bytes())
    }

    /// Compute hash for combining multiple keys.
    #[inline]
    pub fn combine_hashes(h1: u64, h2: u64) -> u64 {
        // Use boost::hash_combine technique
        h1 ^ (h2.wrapping_add(0x9e3779b9).wrapping_add(h1 << 6).wrapping_add(h1 >> 2))
    }
}

/// Open-addressing hash table with linear probing.
#[derive(Debug)]
pub struct HashTable<K, V> {
    /// Keys storage.
    keys: Vec<Option<K>>,
    /// Values storage.
    values: Vec<Option<V>>,
    /// Capacity (always power of 2).
    capacity: usize,
    /// Mask for indexing (capacity - 1).
    mask: usize,
    /// Number of entries.
    len: usize,
    /// Maximum load factor before resize.
    max_load_factor: f64,
}

impl<K: Clone + Eq + Hash, V: Clone> HashTable<K, V> {
    /// Create a new hash table with specified capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        Self {
            keys: vec![None; capacity],
            values: vec![None; capacity],
            capacity,
            mask: capacity - 1,
            len: 0,
            max_load_factor: 0.7,
        }
    }

    /// Create with default capacity.
    pub fn with_default_capacity() -> Self {
        Self::new(64)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get load factor.
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity as f64
    }

    /// Hash a key to an index.
    fn hash_to_index(&self, key: &K) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize & self.mask
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.load_factor() > self.max_load_factor {
            self.resize();
        }

        let mut idx = self.hash_to_index(&key);
        let mut distance = 0;

        loop {
            match &self.keys[idx] {
                None => {
                    self.keys[idx] = Some(key);
                    self.values[idx] = Some(value);
                    self.len += 1;
                    return None;
                }
                Some(existing) if existing == &key => {
                    let old = self.values[idx].take();
                    self.values[idx] = Some(value);
                    return old;
                }
                Some(_) => {
                    idx = (idx + 1) & self.mask;
                    distance += 1;
                    if distance >= self.capacity {
                        panic!("Hash table is full");
                    }
                }
            }
        }
    }

    /// Get a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut idx = self.hash_to_index(key);
        let mut distance = 0;

        loop {
            match &self.keys[idx] {
                None => return None,
                Some(existing) if existing == key => {
                    return self.values[idx].as_ref();
                }
                Some(_) => {
                    idx = (idx + 1) & self.mask;
                    distance += 1;
                    if distance >= self.capacity {
                        return None;
                    }
                }
            }
        }
    }

    /// Get mutable value by key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let mut idx = self.hash_to_index(key);
        let mut distance = 0;

        loop {
            match &self.keys[idx] {
                None => return None,
                Some(existing) if existing == key => {
                    return self.values[idx].as_mut();
                }
                Some(_) => {
                    idx = (idx + 1) & self.mask;
                    distance += 1;
                    if distance >= self.capacity {
                        return None;
                    }
                }
            }
        }
    }

    /// Check if key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get or insert with default.
    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> &mut V
    where
        F: FnOnce() -> V,
    {
        if self.load_factor() > self.max_load_factor {
            self.resize();
        }

        let mut idx = self.hash_to_index(&key);
        let mut distance = 0;

        loop {
            match &self.keys[idx] {
                None => {
                    self.keys[idx] = Some(key);
                    self.values[idx] = Some(f());
                    self.len += 1;
                    return self.values[idx].as_mut().unwrap();
                }
                Some(existing) if existing == &key => {
                    return self.values[idx].as_mut().unwrap();
                }
                Some(_) => {
                    idx = (idx + 1) & self.mask;
                    distance += 1;
                    if distance >= self.capacity {
                        panic!("Hash table is full");
                    }
                }
            }
        }
    }

    /// Resize the hash table.
    fn resize(&mut self) {
        let new_capacity = self.capacity * 2;
        let mut new_keys = vec![None; new_capacity];
        let mut new_values = vec![None; new_capacity];
        let new_mask = new_capacity - 1;

        for i in 0..self.capacity {
            if let Some(key) = self.keys[i].take() {
                let value = self.values[i].take().unwrap();

                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                key.hash(&mut hasher);
                let mut idx = hasher.finish() as usize & new_mask;

                loop {
                    if new_keys[idx].is_none() {
                        new_keys[idx] = Some(key);
                        new_values[idx] = Some(value);
                        break;
                    }
                    idx = (idx + 1) & new_mask;
                }
            }
        }

        self.keys = new_keys;
        self.values = new_values;
        self.capacity = new_capacity;
        self.mask = new_mask;
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.keys.iter().zip(self.values.iter()).filter_map(|(k, v)| {
            k.as_ref().and_then(|key| v.as_ref().map(|val| (key, val)))
        })
    }

    /// Clear the hash table.
    pub fn clear(&mut self) {
        self.keys.fill(None);
        self.values.fill(None);
        self.len = 0;
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Default for HashTable<K, V> {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// Aggregate state for hash aggregation.
#[derive(Debug, Clone, Default)]
pub struct AggState {
    /// Running sum.
    pub sum: f64,
    /// Count of values.
    pub count: usize,
    /// Minimum value seen.
    pub min: f64,
    /// Maximum value seen.
    pub max: f64,
}

impl AggState {
    /// Create new aggregate state.
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Update state with a new value.
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Merge with another state.
    pub fn merge(&mut self, other: &AggState) {
        self.sum += other.sum;
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Get average.
    pub fn avg(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.sum / self.count as f64)
        } else {
            None
        }
    }
}

/// Hash-based GROUP BY aggregator.
#[derive(Debug)]
pub struct HashAggregatorSimd {
    /// Hash table mapping keys to aggregate states.
    table: HashTable<i64, AggState>,
}

impl HashAggregatorSimd {
    /// Create new hash aggregator.
    pub fn new() -> Self {
        Self {
            table: HashTable::new(1024),
        }
    }

    /// Create with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: HashTable::new(capacity),
        }
    }

    /// Aggregate values by keys.
    pub fn aggregate(&mut self, keys: &[i64], values: &[f64]) -> Result<()> {
        if keys.len() != values.len() {
            return Err(Error::DimensionMismatch(format!(
                "Keys length {} != values length {}",
                keys.len(),
                values.len()
            )));
        }

        // Process in blocks for cache efficiency
        for (key_chunk, val_chunk) in keys.chunks(BLOCK_SIZE).zip(values.chunks(BLOCK_SIZE)) {
            for (&key, &value) in key_chunk.iter().zip(val_chunk.iter()) {
                let state = self.table.get_or_insert_with(key, AggState::new);
                state.update(value);
            }
        }

        Ok(())
    }

    /// Aggregate i32 keys with f64 values.
    pub fn aggregate_i32(&mut self, keys: &[i32], values: &[f64]) -> Result<()> {
        if keys.len() != values.len() {
            return Err(Error::DimensionMismatch(format!(
                "Keys length {} != values length {}",
                keys.len(),
                values.len()
            )));
        }

        for (&key, &value) in keys.iter().zip(values.iter()) {
            let state = self.table.get_or_insert_with(key as i64, AggState::new);
            state.update(value);
        }

        Ok(())
    }

    /// Get number of groups.
    pub fn num_groups(&self) -> usize {
        self.table.len()
    }

    /// Get aggregate state for a key.
    pub fn get(&self, key: i64) -> Option<&AggState> {
        self.table.get(&key)
    }

    /// Get sum for a key.
    pub fn get_sum(&self, key: i64) -> Option<f64> {
        self.table.get(&key).map(|s| s.sum)
    }

    /// Get count for a key.
    pub fn get_count(&self, key: i64) -> Option<usize> {
        self.table.get(&key).map(|s| s.count)
    }

    /// Get average for a key.
    pub fn get_avg(&self, key: i64) -> Option<f64> {
        self.table.get(&key).and_then(|s| s.avg())
    }

    /// Get min for a key.
    pub fn get_min(&self, key: i64) -> Option<f64> {
        self.table.get(&key).map(|s| s.min)
    }

    /// Get max for a key.
    pub fn get_max(&self, key: i64) -> Option<f64> {
        self.table.get(&key).map(|s| s.max)
    }

    /// Get all results as vectors.
    pub fn get_results(&self) -> (Vec<i64>, Vec<AggState>) {
        let mut keys = Vec::with_capacity(self.table.len());
        let mut states = Vec::with_capacity(self.table.len());

        for (key, state) in self.table.iter() {
            keys.push(*key);
            states.push(state.clone());
        }

        (keys, states)
    }

    /// Clear the aggregator.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl Default for HashAggregatorSimd {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash join implementation.
#[derive(Debug)]
pub struct HashJoin<V: Clone> {
    /// Build side hash table.
    build_table: HashTable<i64, Vec<V>>,
}

impl<V: Clone> HashJoin<V> {
    /// Create new hash join.
    pub fn new() -> Self {
        Self {
            build_table: HashTable::new(1024),
        }
    }

    /// Build phase: insert (key, value) pairs into hash table.
    pub fn build(&mut self, keys: &[i64], values: &[V]) -> Result<()> {
        if keys.len() != values.len() {
            return Err(Error::DimensionMismatch(format!(
                "Keys length {} != values length {}",
                keys.len(),
                values.len()
            )));
        }

        for (&key, value) in keys.iter().zip(values.iter()) {
            let entry = self.build_table.get_or_insert_with(key, Vec::new);
            entry.push(value.clone());
        }

        Ok(())
    }

    /// Probe phase: find matching values for probe keys.
    pub fn probe(&self, probe_keys: &[i64]) -> Vec<Option<Vec<V>>> {
        probe_keys
            .iter()
            .map(|key| self.build_table.get(key).cloned())
            .collect()
    }

    /// Probe single key.
    pub fn probe_single(&self, key: i64) -> Option<&Vec<V>> {
        self.build_table.get(&key)
    }

    /// Get build table size.
    pub fn build_size(&self) -> usize {
        self.build_table.len()
    }

    /// Clear the join.
    pub fn clear(&mut self) {
        self.build_table.clear();
    }
}

impl<V: Clone> Default for HashJoin<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Bloom filter for approximate membership testing.
#[derive(Debug)]
pub struct BloomFilter {
    /// Bit array.
    bits: Vec<u64>,
    /// Number of hash functions.
    num_hashes: usize,
    /// Number of bits.
    num_bits: usize,
}

impl BloomFilter {
    /// Create a new bloom filter.
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal parameters
        let num_bits = Self::optimal_bits(expected_items, false_positive_rate);
        let num_hashes = Self::optimal_hashes(num_bits, expected_items);
        let num_words = (num_bits + 63) / 64;

        Self {
            bits: vec![0; num_words],
            num_hashes,
            num_bits,
        }
    }

    /// Calculate optimal number of bits.
    fn optimal_bits(n: usize, p: f64) -> usize {
        let m = -(n as f64 * p.ln()) / (2.0_f64.ln().powi(2));
        (m as usize).max(64)
    }

    /// Calculate optimal number of hash functions.
    fn optimal_hashes(m: usize, n: usize) -> usize {
        let k = (m as f64 / n as f64) * 2.0_f64.ln();
        (k as usize).max(1).min(16)
    }

    /// Insert a value.
    pub fn insert(&mut self, value: i64) {
        for i in 0..self.num_hashes {
            let hash = self.hash_nth(value, i);
            let bit_idx = hash % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_offset;
        }
    }

    /// Check if value might be present.
    pub fn might_contain(&self, value: i64) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash_nth(value, i);
            let bit_idx = hash % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            if self.bits[word_idx] & (1u64 << bit_offset) == 0 {
                return false;
            }
        }
        true
    }

    /// Get nth hash of value.
    fn hash_nth(&self, value: i64, n: usize) -> usize {
        let h1 = VectorizedHash::hash_i64(value);
        let h2 = VectorizedHash::hash_i64(value.wrapping_add(n as i64));
        (h1.wrapping_add((n as u64).wrapping_mul(h2))) as usize
    }

    /// Get approximate false positive rate.
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits: u32 = self.bits.iter().map(|w| w.count_ones()).sum();
        let p = set_bits as f64 / self.num_bits as f64;
        p.powi(self.num_hashes as i32)
    }

    /// Clear the filter.
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
}

/// Count-min sketch for frequency estimation.
#[derive(Debug)]
pub struct CountMinSketch {
    /// 2D array of counters.
    counters: Vec<Vec<u64>>,
    /// Width of each row.
    width: usize,
    /// Number of rows (hash functions).
    depth: usize,
}

impl CountMinSketch {
    /// Create a new count-min sketch.
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            counters: vec![vec![0; width]; depth],
            width,
            depth,
        }
    }

    /// Create with error bounds.
    pub fn with_error_bounds(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width.max(16), depth.max(2))
    }

    /// Increment count for a value.
    pub fn increment(&mut self, value: i64) {
        for i in 0..self.depth {
            let hash = self.hash_nth(value, i);
            self.counters[i][hash] = self.counters[i][hash].saturating_add(1);
        }
    }

    /// Increment by amount.
    pub fn increment_by(&mut self, value: i64, amount: u64) {
        for i in 0..self.depth {
            let hash = self.hash_nth(value, i);
            self.counters[i][hash] = self.counters[i][hash].saturating_add(amount);
        }
    }

    /// Estimate count for a value.
    pub fn estimate(&self, value: i64) -> u64 {
        (0..self.depth)
            .map(|i| {
                let hash = self.hash_nth(value, i);
                self.counters[i][hash]
            })
            .min()
            .unwrap_or(0)
    }

    /// Get nth hash of value.
    fn hash_nth(&self, value: i64, n: usize) -> usize {
        let h1 = VectorizedHash::hash_i64(value);
        let h2 = VectorizedHash::hash_i64(value.wrapping_add((n as i64).wrapping_mul(0x517cc1b727220a95_u64 as i64)));
        (h1.wrapping_add((n as u64).wrapping_mul(h2)) as usize) % self.width
    }

    /// Clear the sketch.
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
    }
}

/// HyperLogLog for cardinality estimation.
#[derive(Debug)]
pub struct HyperLogLog {
    /// Registers.
    registers: Vec<u8>,
    /// Number of registers (2^precision).
    num_registers: usize,
    /// Precision bits.
    precision: u8,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with specified precision.
    pub fn new(precision: u8) -> Self {
        let precision = precision.clamp(4, 16);
        let num_registers = 1 << precision;
        Self {
            registers: vec![0; num_registers],
            num_registers,
            precision,
        }
    }

    /// Add a value.
    pub fn add(&mut self, value: i64) {
        let hash = VectorizedHash::hash_i64(value);
        let idx = (hash >> (64 - self.precision)) as usize;
        let remaining = hash << self.precision | (1u64 << (self.precision - 1));
        let rank = (remaining.leading_zeros() + 1) as u8;
        self.registers[idx] = self.registers[idx].max(rank);
    }

    /// Estimate cardinality.
    pub fn estimate(&self) -> f64 {
        let m = self.num_registers as f64;
        let alpha = self.get_alpha();

        let sum: f64 = self.registers.iter().map(|&r| 2.0_f64.powi(-(r as i32))).sum();
        let estimate = alpha * m * m / sum;

        // Small range correction
        if estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }

        estimate
    }

    /// Get alpha constant for precision.
    fn get_alpha(&self) -> f64 {
        match self.num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / self.num_registers as f64),
        }
    }

    /// Merge with another HyperLogLog.
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(self.precision, other.precision);
        for i in 0..self.num_registers {
            self.registers[i] = self.registers[i].max(other.registers[i]);
        }
    }

    /// Clear the estimator.
    pub fn clear(&mut self) {
        self.registers.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorized_hash_i64() {
        let h1 = VectorizedHash::hash_i64(42);
        let h2 = VectorizedHash::hash_i64(42);
        assert_eq!(h1, h2);

        let h3 = VectorizedHash::hash_i64(43);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_table_basic() {
        let mut table: HashTable<i64, f64> = HashTable::new(16);

        table.insert(1, 10.0);
        table.insert(2, 20.0);
        table.insert(3, 30.0);

        assert_eq!(table.len(), 3);
        assert_eq!(table.get(&1), Some(&10.0));
        assert_eq!(table.get(&2), Some(&20.0));
        assert_eq!(table.get(&3), Some(&30.0));
        assert_eq!(table.get(&4), None);
    }

    #[test]
    fn test_hash_table_update() {
        let mut table: HashTable<i64, i32> = HashTable::new(16);

        assert_eq!(table.insert(1, 100), None);
        assert_eq!(table.insert(1, 200), Some(100));
        assert_eq!(table.get(&1), Some(&200));
    }

    #[test]
    fn test_hash_table_resize() {
        let mut table: HashTable<i64, i32> = HashTable::new(8);

        for i in 0..100 {
            table.insert(i, i as i32 * 10);
        }

        assert_eq!(table.len(), 100);
        for i in 0..100 {
            assert_eq!(table.get(&i), Some(&(i as i32 * 10)));
        }
    }

    #[test]
    fn test_hash_aggregator() {
        let mut agg = HashAggregatorSimd::new();

        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        agg.aggregate(&keys, &values).unwrap();

        assert_eq!(agg.num_groups(), 2);
        assert_eq!(agg.get_sum(1), Some(90.0)); // 10 + 30 + 50
        assert_eq!(agg.get_sum(2), Some(60.0)); // 20 + 40
        assert_eq!(agg.get_count(1), Some(3));
        assert_eq!(agg.get_count(2), Some(2));
    }

    #[test]
    fn test_hash_join() {
        let mut join: HashJoin<String> = HashJoin::new();

        let build_keys = vec![1, 2, 3];
        let build_values = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        join.build(&build_keys, &build_values).unwrap();

        let probe_keys = vec![1, 2, 4];
        let results = join.probe(&probe_keys);

        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1000, 0.01);

        for i in 0..100 {
            filter.insert(i);
        }

        // All inserted elements should be found
        for i in 0..100 {
            assert!(filter.might_contain(i));
        }

        // Most non-inserted elements should not be found
        let mut false_positives = 0;
        for i in 100..1000 {
            if filter.might_contain(i) {
                false_positives += 1;
            }
        }
        assert!(false_positives < 50); // Less than 5% false positive
    }

    #[test]
    fn test_count_min_sketch() {
        let mut sketch = CountMinSketch::new(1000, 5);

        sketch.increment(1);
        sketch.increment(1);
        sketch.increment(1);
        sketch.increment(2);

        assert_eq!(sketch.estimate(1), 3);
        assert_eq!(sketch.estimate(2), 1);
        assert_eq!(sketch.estimate(3), 0);
    }

    #[test]
    fn test_hyperloglog() {
        let mut hll = HyperLogLog::new(10);

        for i in 0..10000 {
            hll.add(i);
        }

        let estimate = hll.estimate();
        // HLL should be within ~3% of actual
        assert!((estimate - 10000.0).abs() < 500.0);
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog::new(10);
        let mut hll2 = HyperLogLog::new(10);

        for i in 0..5000 {
            hll1.add(i);
        }
        for i in 2500..7500 {
            hll2.add(i);
        }

        hll1.merge(&hll2);
        let estimate = hll1.estimate();
        // Should estimate ~7500 unique values
        assert!((estimate - 7500.0).abs() < 400.0);
    }
}
