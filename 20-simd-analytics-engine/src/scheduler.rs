//! Parallel execution scheduler.

use crate::aggregate::{AggregateOp, AggregateValue, PartialAggregate};
use crate::filter::{FilterOp, SelectionVector, VectorizedFilter};
use crate::{Error, Result, BLOCK_SIZE};
use rayon::prelude::*;
use std::sync::Arc;

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Batch size for parallel processing.
    pub batch_size: usize,
    /// Enable parallel execution.
    pub parallel: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            batch_size: BLOCK_SIZE,
            parallel: true,
        }
    }
}

/// Parallel executor for analytics operations.
pub struct ParallelExecutor {
    config: ExecutorConfig,
}

impl ParallelExecutor {
    /// Create new executor.
    pub fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Parallel sum of f64 data.
    pub fn parallel_sum_f64(&self, data: &[f64]) -> f64 {
        if !self.config.parallel || data.len() < self.config.batch_size {
            return data.iter().sum();
        }

        data.par_chunks(self.config.batch_size)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum()
    }

    /// Parallel aggregate of f64 data.
    pub fn parallel_aggregate_f64(
        &self,
        data: &[f64],
        op: AggregateOp,
    ) -> AggregateValue {
        if data.is_empty() {
            return AggregateValue::Null;
        }

        if !self.config.parallel || data.len() < self.config.batch_size {
            return match op {
                AggregateOp::Sum => AggregateValue::Float64(data.iter().sum()),
                AggregateOp::Count => AggregateValue::Count(data.len()),
                AggregateOp::Min => {
                    AggregateValue::Float64(data.iter().cloned().fold(f64::INFINITY, f64::min))
                }
                AggregateOp::Max => {
                    AggregateValue::Float64(data.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                }
                AggregateOp::Avg => {
                    let sum: f64 = data.iter().sum();
                    AggregateValue::Float64(sum / data.len() as f64)
                }
            };
        }

        // Compute partial aggregates in parallel
        let partials: Vec<PartialAggregate> = data
            .par_chunks(self.config.batch_size)
            .map(PartialAggregate::from_slice)
            .collect();

        // Merge partials
        let mut merged = PartialAggregate::default();
        for partial in &partials {
            merged.merge(partial);
        }

        merged.finalize(op)
    }

    /// Parallel filter of f64 data.
    pub fn parallel_filter_f64(
        &self,
        data: &[f64],
        op: FilterOp,
        threshold: f64,
    ) -> SelectionVector {
        if !self.config.parallel || data.len() < self.config.batch_size {
            return VectorizedFilter::filter_f64(data, op, threshold);
        }

        let num_chunks = (data.len() + self.config.batch_size - 1) / self.config.batch_size;

        // Process chunks in parallel
        let chunk_selections: Vec<(usize, SelectionVector)> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(data.len());
                let chunk = &data[start..end];
                let selection = VectorizedFilter::filter_f64(chunk, op, threshold);
                (start, selection)
            })
            .collect();

        // Merge selections
        let mut result = SelectionVector::new(data.len());
        for (start, selection) in chunk_selections {
            for i in 0..selection.num_rows() {
                result.set(start + i, selection.is_selected(i));
            }
        }
        result.recount();
        result
    }

    /// Parallel filter of f32 data.
    pub fn parallel_filter_f32(
        &self,
        data: &[f32],
        op: FilterOp,
        threshold: f32,
    ) -> SelectionVector {
        if !self.config.parallel || data.len() < self.config.batch_size {
            return VectorizedFilter::filter_f32(data, op, threshold);
        }

        let num_chunks = (data.len() + self.config.batch_size - 1) / self.config.batch_size;

        let chunk_selections: Vec<(usize, SelectionVector)> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(data.len());
                let chunk = &data[start..end];
                let selection = VectorizedFilter::filter_f32(chunk, op, threshold);
                (start, selection)
            })
            .collect();

        let mut result = SelectionVector::new(data.len());
        for (start, selection) in chunk_selections {
            for i in 0..selection.num_rows() {
                result.set(start + i, selection.is_selected(i));
            }
        }
        result.recount();
        result
    }

    /// Parallel map operation.
    pub fn parallel_map_f64<F>(&self, data: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        if !self.config.parallel || data.len() < self.config.batch_size {
            return data.iter().map(|&x| f(x)).collect();
        }

        data.par_iter().map(|&x| f(x)).collect()
    }

    /// Parallel scalar multiplication.
    pub fn parallel_scale_f64(&self, data: &[f64], scalar: f64) -> Vec<f64> {
        self.parallel_map_f64(data, |x| x * scalar)
    }

    /// Parallel element-wise addition.
    pub fn parallel_add_f64(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().min(b.len());
        if !self.config.parallel || len < self.config.batch_size {
            return a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        }

        (0..len)
            .into_par_iter()
            .map(|i| a[i] + b[i])
            .collect()
    }

    /// Parallel dot product.
    pub fn parallel_dot_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        if !self.config.parallel || len < self.config.batch_size {
            return a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        }

        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    /// Parallel count matching predicate.
    pub fn parallel_count_if<F>(&self, data: &[f64], predicate: F) -> usize
    where
        F: Fn(f64) -> bool + Send + Sync,
    {
        if !self.config.parallel || data.len() < self.config.batch_size {
            return data.iter().filter(|&&x| predicate(x)).count();
        }

        data.par_iter().filter(|&&x| predicate(x)).count()
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ExecutorConfig::default())
    }
}

/// Partition data for parallel processing.
pub struct Partitioner;

impl Partitioner {
    /// Partition data into roughly equal chunks.
    pub fn partition<T: Clone>(data: &[T], num_partitions: usize) -> Vec<Vec<T>> {
        if num_partitions == 0 || data.is_empty() {
            return vec![];
        }

        let chunk_size = (data.len() + num_partitions - 1) / num_partitions;
        data.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Hash partition by key.
    pub fn hash_partition<K: std::hash::Hash, V: Clone>(
        keys: &[K],
        values: &[V],
        num_partitions: usize,
    ) -> Vec<Vec<V>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut partitions: Vec<Vec<V>> = (0..num_partitions).map(|_| Vec::new()).collect();

        for (key, value) in keys.iter().zip(values.iter()) {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let partition = (hasher.finish() as usize) % num_partitions;
            partitions[partition].push(value.clone());
        }

        partitions
    }
}

/// Work stealing task queue for dynamic load balancing.
pub struct WorkQueue<T> {
    tasks: parking_lot::Mutex<Vec<T>>,
}

impl<T> WorkQueue<T> {
    /// Create new work queue.
    pub fn new() -> Self {
        Self {
            tasks: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Add task.
    pub fn push(&self, task: T) {
        self.tasks.lock().push(task);
    }

    /// Take task.
    pub fn pop(&self) -> Option<T> {
        self.tasks.lock().pop()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.lock().is_empty()
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.tasks.lock().len()
    }
}

impl<T> Default for WorkQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline stage for streaming execution.
pub trait PipelineStage: Send + Sync {
    /// Process a batch of data.
    fn process(&self, input: Vec<f64>) -> Result<Vec<f64>>;

    /// Get stage name.
    fn name(&self) -> &str;
}

/// Filter pipeline stage.
pub struct FilterStage {
    op: FilterOp,
    threshold: f64,
}

impl FilterStage {
    pub fn new(op: FilterOp, threshold: f64) -> Self {
        Self { op, threshold }
    }
}

impl PipelineStage for FilterStage {
    fn process(&self, input: Vec<f64>) -> Result<Vec<f64>> {
        let selection = VectorizedFilter::filter_f64(&input, self.op, self.threshold);
        let mut output = Vec::with_capacity(selection.count());
        for (i, &value) in input.iter().enumerate() {
            if selection.is_selected(i) {
                output.push(value);
            }
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "Filter"
    }
}

/// Map pipeline stage.
pub struct MapStage<F>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    func: F,
}

impl<F> MapStage<F>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> PipelineStage for MapStage<F>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    fn process(&self, input: Vec<f64>) -> Result<Vec<f64>> {
        Ok(input.into_iter().map(&self.func).collect())
    }

    fn name(&self) -> &str {
        "Map"
    }
}

/// Pipeline executor.
pub struct Pipeline {
    stages: Vec<Arc<dyn PipelineStage>>,
}

impl Pipeline {
    /// Create new pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add stage to pipeline.
    pub fn add_stage<S: PipelineStage + 'static>(&mut self, stage: S) {
        self.stages.push(Arc::new(stage));
    }

    /// Execute pipeline on data.
    pub fn execute(&self, input: Vec<f64>) -> Result<Vec<f64>> {
        let mut data = input;
        for stage in &self.stages {
            data = stage.process(data)?;
        }
        Ok(data)
    }

    /// Execute pipeline in batches.
    pub fn execute_batched(&self, input: Vec<f64>, batch_size: usize) -> Result<Vec<f64>> {
        let batches: Vec<Vec<f64>> = input
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let results: Result<Vec<Vec<f64>>> = batches
            .into_par_iter()
            .map(|batch| self.execute(batch))
            .collect();

        Ok(results?.into_iter().flatten().collect())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sum() {
        let executor = ParallelExecutor::default();
        let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();

        let sum = executor.parallel_sum_f64(&data);
        let expected: f64 = (0..10000).map(|i| i as f64).sum();

        assert!((sum - expected).abs() < 0.001);
    }

    #[test]
    fn test_parallel_aggregate() {
        let executor = ParallelExecutor::default();
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        let sum = executor.parallel_aggregate_f64(&data, AggregateOp::Sum);
        assert_eq!(sum.as_f64(), Some(5050.0));

        let min = executor.parallel_aggregate_f64(&data, AggregateOp::Min);
        assert_eq!(min.as_f64(), Some(1.0));

        let max = executor.parallel_aggregate_f64(&data, AggregateOp::Max);
        assert_eq!(max.as_f64(), Some(100.0));
    }

    #[test]
    fn test_parallel_filter() {
        let executor = ParallelExecutor::default();
        let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();

        let selection = executor.parallel_filter_f64(&data, FilterOp::Gt, 5000.0);
        assert_eq!(selection.count(), 4999); // 5001..9999
    }

    #[test]
    fn test_partitioner() {
        let data: Vec<i32> = (0..100).collect();
        let partitions = Partitioner::partition(&data, 4);

        assert_eq!(partitions.len(), 4);
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = Pipeline::new();
        pipeline.add_stage(FilterStage::new(FilterOp::Gt, 3.0));
        pipeline.add_stage(MapStage::new(|x| x * 2.0));

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = pipeline.execute(input).unwrap();

        assert_eq!(output, vec![8.0, 10.0]); // 4, 5 -> 8, 10
    }
}
