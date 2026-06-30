//! Kernel autotuning for finding optimal configurations.

use crate::gemm::{GemmConfig, GemmKernel};
use crate::matrix::Matrix;
use crate::metrics::{Benchmark, GemmMetrics};
use crate::{Error, Result};
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

/// Autotuning configuration.
#[derive(Debug, Clone)]
pub struct AutotuneConfig {
    /// Search strategy.
    pub strategy: SearchStrategy,
    /// Maximum number of configurations to try.
    pub max_trials: usize,
    /// Time budget for tuning.
    pub time_budget: Option<Duration>,
    /// Early stopping threshold (stop if no improvement for N trials).
    pub early_stop: usize,
    /// Number of benchmark iterations per configuration.
    pub benchmark_iters: usize,
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::GridSearch,
            max_trials: 100,
            time_budget: None,
            early_stop: 20,
            benchmark_iters: 5,
        }
    }
}

/// Search strategy for autotuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Exhaustive grid search.
    GridSearch,
    /// Random search.
    Random,
    /// Simulated annealing.
    SimulatedAnnealing,
    /// Genetic algorithm.
    Genetic,
}

/// Parameter space for GEMM configuration.
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Block M sizes to try.
    pub block_m: Vec<usize>,
    /// Block N sizes to try.
    pub block_n: Vec<usize>,
    /// Block K sizes to try.
    pub block_k: Vec<usize>,
    /// Thread M sizes to try.
    pub thread_m: Vec<usize>,
    /// Thread N sizes to try.
    pub thread_n: Vec<usize>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            block_m: vec![32, 64, 128, 256],
            block_n: vec![32, 64, 128, 256],
            block_k: vec![4, 8, 16, 32],
            thread_m: vec![4, 8, 16],
            thread_n: vec![4, 8, 16],
        }
    }
}

impl ParameterSpace {
    /// Create parameter space optimized for small matrices.
    pub fn small() -> Self {
        Self {
            block_m: vec![16, 32, 64],
            block_n: vec![16, 32, 64],
            block_k: vec![4, 8, 16],
            thread_m: vec![4, 8],
            thread_n: vec![4, 8],
        }
    }

    /// Create parameter space optimized for large matrices.
    pub fn large() -> Self {
        Self {
            block_m: vec![64, 128, 256],
            block_n: vec![64, 128, 256],
            block_k: vec![8, 16, 32],
            thread_m: vec![8, 16],
            thread_n: vec![8, 16],
        }
    }

    /// Generate all valid configurations.
    pub fn generate_configs(&self) -> Vec<GemmConfig> {
        let mut configs = Vec::new();

        for &bm in &self.block_m {
            for &bn in &self.block_n {
                for &bk in &self.block_k {
                    for &tm in &self.thread_m {
                        for &tn in &self.thread_n {
                            let config = GemmConfig {
                                block_m: bm,
                                block_n: bn,
                                block_k: bk,
                                thread_m: tm,
                                thread_n: tn,
                            };

                            // Only include valid configurations
                            if config.validate().is_ok() {
                                configs.push(config);
                            }
                        }
                    }
                }
            }
        }

        configs
    }

    /// Total number of valid configurations.
    pub fn num_configs(&self) -> usize {
        self.generate_configs().len()
    }
}

/// Result of autotuning.
#[derive(Debug, Clone)]
pub struct AutotuneResult {
    /// Best configuration found.
    pub best_config: GemmConfig,
    /// Metrics for best configuration.
    pub best_metrics: GemmMetrics,
    /// Number of configurations tried.
    pub num_trials: usize,
    /// Total tuning time.
    pub total_time: Duration,
    /// History of all tried configurations.
    pub history: Vec<(GemmConfig, GemmMetrics)>,
}

impl AutotuneResult {
    /// Format results as string.
    pub fn format(&self) -> String {
        format!(
            "Autotuning Results:\n\
             \x20 Best Config: block_m={}, block_n={}, block_k={}, thread_m={}, thread_n={}\n\
             \x20 Performance: {:.2} GFLOPS\n\
             \x20 Trials: {} in {:.2}s",
            self.best_config.block_m,
            self.best_config.block_n,
            self.best_config.block_k,
            self.best_config.thread_m,
            self.best_config.thread_n,
            self.best_metrics.gflops,
            self.num_trials,
            self.total_time.as_secs_f64()
        )
    }
}

/// Kernel autotuner.
pub struct Autotuner {
    /// Autotuning configuration.
    pub config: AutotuneConfig,
    /// Parameter space.
    pub param_space: ParameterSpace,
    /// Benchmark runner.
    pub benchmark: Benchmark,
}

impl Autotuner {
    /// Create new autotuner.
    pub fn new(config: AutotuneConfig, param_space: ParameterSpace) -> Self {
        let benchmark = Benchmark::new(2, config.benchmark_iters, 100.0);
        Self {
            config,
            param_space,
            benchmark,
        }
    }

    /// Run autotuning with given kernel.
    pub fn tune<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        match self.config.strategy {
            SearchStrategy::GridSearch => self.grid_search(a, b, kernel_fn),
            SearchStrategy::Random => self.random_search(a, b, kernel_fn),
            SearchStrategy::SimulatedAnnealing => self.simulated_annealing(a, b, kernel_fn),
            SearchStrategy::Genetic => self.genetic_search(a, b, kernel_fn),
        }
    }

    /// Grid search over all configurations.
    fn grid_search<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let configs = self.param_space.generate_configs();

        let mut best_config = GemmConfig::default();
        let mut best_metrics: Option<GemmMetrics> = None;
        let mut history = Vec::new();
        let mut no_improvement = 0;

        let max_trials = self.config.max_trials.min(configs.len());

        for (i, config) in configs.into_iter().take(max_trials).enumerate() {
            // Check time budget
            if let Some(budget) = self.config.time_budget {
                if start.elapsed() > budget {
                    break;
                }
            }

            // Benchmark this configuration
            let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
                kernel_fn(a_mat, b_mat, c_mat, &config)
            };

            match self.benchmark.run(a, b, kernel) {
                Ok(metrics) => {
                    let is_better = match &best_metrics {
                        None => true,
                        Some(best) => metrics.gflops > best.gflops,
                    };

                    if is_better {
                        best_config = config;
                        best_metrics = Some(metrics.clone());
                        no_improvement = 0;
                    } else {
                        no_improvement += 1;
                    }

                    history.push((config, metrics));
                }
                Err(_) => continue, // Skip invalid configurations
            }

            // Early stopping
            if no_improvement >= self.config.early_stop {
                break;
            }
        }

        let best_metrics = best_metrics
            .ok_or_else(|| Error::InvalidConfig("No valid configuration found".into()))?;

        Ok(AutotuneResult {
            best_config,
            best_metrics,
            num_trials: history.len(),
            total_time: start.elapsed(),
            history,
        })
    }

    /// Random search over configurations.
    fn random_search<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();
        let configs = self.param_space.generate_configs();

        let mut best_config = GemmConfig::default();
        let mut best_metrics: Option<GemmMetrics> = None;
        let mut history = Vec::new();
        let mut no_improvement = 0;

        for _ in 0..self.config.max_trials {
            // Check time budget
            if let Some(budget) = self.config.time_budget {
                if start.elapsed() > budget {
                    break;
                }
            }

            // Pick random configuration
            let config = configs.choose(&mut rng).unwrap().clone();

            // Benchmark
            let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
                kernel_fn(a_mat, b_mat, c_mat, &config)
            };

            match self.benchmark.run(a, b, kernel) {
                Ok(metrics) => {
                    let is_better = match &best_metrics {
                        None => true,
                        Some(best) => metrics.gflops > best.gflops,
                    };

                    if is_better {
                        best_config = config;
                        best_metrics = Some(metrics.clone());
                        no_improvement = 0;
                    } else {
                        no_improvement += 1;
                    }

                    history.push((config, metrics));
                }
                Err(_) => continue,
            }

            if no_improvement >= self.config.early_stop {
                break;
            }
        }

        let best_metrics = best_metrics
            .ok_or_else(|| Error::InvalidConfig("No valid configuration found".into()))?;

        Ok(AutotuneResult {
            best_config,
            best_metrics,
            num_trials: history.len(),
            total_time: start.elapsed(),
            history,
        })
    }

    /// Simulated annealing search.
    fn simulated_annealing<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        // Start with default configuration
        let mut current = GemmConfig::default();
        let mut current_score = 0.0f64;
        let mut best_config = current;
        let mut best_score = 0.0f64;
        let mut history = Vec::new();

        // Initial evaluation
        let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
            kernel_fn(a_mat, b_mat, c_mat, &current)
        };
        if let Ok(metrics) = self.benchmark.run(a, b, kernel) {
            current_score = metrics.gflops;
            best_score = current_score;
            history.push((current, metrics));
        }

        // Annealing parameters
        let mut temperature = 1.0;
        let cooling_rate = 0.95;
        let min_temperature = 0.01;

        for _ in 0..self.config.max_trials {
            if let Some(budget) = self.config.time_budget {
                if start.elapsed() > budget {
                    break;
                }
            }

            if temperature < min_temperature {
                break;
            }

            // Generate neighbor by mutating one parameter
            let neighbor = self.mutate_config(&current, &mut rng);

            // Evaluate neighbor
            let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
                kernel_fn(a_mat, b_mat, c_mat, &neighbor)
            };

            if let Ok(metrics) = self.benchmark.run(a, b, kernel) {
                let neighbor_score = metrics.gflops;
                let delta = neighbor_score - current_score;

                // Accept if better or with probability based on temperature
                let accept = if delta > 0.0 {
                    true
                } else {
                    let prob = (delta / temperature).exp();
                    rng.gen::<f64>() < prob
                };

                if accept {
                    current = neighbor;
                    current_score = neighbor_score;

                    if current_score > best_score {
                        best_config = current;
                        best_score = current_score;
                    }
                }

                history.push((neighbor, metrics));
            }

            temperature *= cooling_rate;
        }

        // Get best metrics
        let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
            kernel_fn(a_mat, b_mat, c_mat, &best_config)
        };
        let best_metrics = self.benchmark.run(a, b, kernel)?;

        Ok(AutotuneResult {
            best_config,
            best_metrics,
            num_trials: history.len(),
            total_time: start.elapsed(),
            history,
        })
    }

    /// Genetic algorithm search.
    fn genetic_search<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        let population_size = 20;
        let elite_count = 4;
        let mutation_rate = 0.1;

        // Initialize population
        let configs = self.param_space.generate_configs();
        let mut population: Vec<GemmConfig> = configs
            .choose_multiple(&mut rng, population_size.min(configs.len()))
            .cloned()
            .collect();

        let mut best_config = GemmConfig::default();
        let mut best_score = 0.0f64;
        let mut history = Vec::new();

        let generations = self.config.max_trials / population_size;

        for _ in 0..generations {
            if let Some(budget) = self.config.time_budget {
                if start.elapsed() > budget {
                    break;
                }
            }

            // Evaluate population
            let mut scored: Vec<(GemmConfig, f64, GemmMetrics)> = Vec::new();

            for config in &population {
                let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
                    kernel_fn(a_mat, b_mat, c_mat, config)
                };

                if let Ok(metrics) = self.benchmark.run(a, b, kernel) {
                    scored.push((config.clone(), metrics.gflops, metrics));
                }
            }

            if scored.is_empty() {
                continue;
            }

            // Sort by fitness
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Update best
            if scored[0].1 > best_score {
                best_config = scored[0].0.clone();
                best_score = scored[0].1;
            }

            // Record history
            for (config, _, metrics) in &scored {
                history.push((config.clone(), metrics.clone()));
            }

            // Selection and reproduction
            let mut new_population = Vec::with_capacity(population_size);

            // Keep elite
            for i in 0..elite_count.min(scored.len()) {
                new_population.push(scored[i].0.clone());
            }

            // Crossover and mutation
            while new_population.len() < population_size {
                // Tournament selection
                let parent1 = self.tournament_select(&scored, &mut rng);
                let parent2 = self.tournament_select(&scored, &mut rng);

                // Crossover
                let mut child = self.crossover(&parent1, &parent2, &mut rng);

                // Mutation
                if rng.gen::<f64>() < mutation_rate {
                    child = self.mutate_config(&child, &mut rng);
                }

                if child.validate().is_ok() {
                    new_population.push(child);
                }
            }

            population = new_population;
        }

        // Get best metrics
        let kernel = |a_mat: &Matrix, b_mat: &Matrix, c_mat: &mut Matrix| {
            kernel_fn(a_mat, b_mat, c_mat, &best_config)
        };
        let best_metrics = self.benchmark.run(a, b, kernel)?;

        Ok(AutotuneResult {
            best_config,
            best_metrics,
            num_trials: history.len(),
            total_time: start.elapsed(),
            history,
        })
    }

    /// Mutate a configuration.
    fn mutate_config(&self, config: &GemmConfig, rng: &mut impl Rng) -> GemmConfig {
        let mut new_config = config.clone();

        match rng.gen_range(0..5) {
            0 => {
                if let Some(&v) = self.param_space.block_m.choose(rng) {
                    new_config.block_m = v;
                }
            }
            1 => {
                if let Some(&v) = self.param_space.block_n.choose(rng) {
                    new_config.block_n = v;
                }
            }
            2 => {
                if let Some(&v) = self.param_space.block_k.choose(rng) {
                    new_config.block_k = v;
                }
            }
            3 => {
                if let Some(&v) = self.param_space.thread_m.choose(rng) {
                    new_config.thread_m = v;
                }
            }
            _ => {
                if let Some(&v) = self.param_space.thread_n.choose(rng) {
                    new_config.thread_n = v;
                }
            }
        }

        // Ensure validity
        if new_config.validate().is_err() {
            return config.clone();
        }

        new_config
    }

    /// Tournament selection for genetic algorithm.
    fn tournament_select(
        &self,
        scored: &[(GemmConfig, f64, GemmMetrics)],
        rng: &mut impl Rng,
    ) -> GemmConfig {
        let tournament_size = 3;
        let mut best_idx = rng.gen_range(0..scored.len());
        let mut best_fitness = scored[best_idx].1;

        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..scored.len());
            if scored[idx].1 > best_fitness {
                best_idx = idx;
                best_fitness = scored[idx].1;
            }
        }

        scored[best_idx].0.clone()
    }

    /// Crossover two configurations.
    fn crossover(&self, parent1: &GemmConfig, parent2: &GemmConfig, rng: &mut impl Rng) -> GemmConfig {
        GemmConfig {
            block_m: if rng.gen() { parent1.block_m } else { parent2.block_m },
            block_n: if rng.gen() { parent1.block_n } else { parent2.block_n },
            block_k: if rng.gen() { parent1.block_k } else { parent2.block_k },
            thread_m: if rng.gen() { parent1.thread_m } else { parent2.thread_m },
            thread_n: if rng.gen() { parent1.thread_n } else { parent2.thread_n },
        }
    }
}

/// Cache for autotuning results.
#[derive(Debug, Default)]
pub struct TuningCache {
    /// Cached results by matrix dimensions.
    cache: HashMap<(usize, usize, usize), GemmConfig>,
}

impl TuningCache {
    /// Create new cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get cached configuration.
    pub fn get(&self, m: usize, n: usize, k: usize) -> Option<&GemmConfig> {
        self.cache.get(&(m, n, k))
    }

    /// Store configuration.
    pub fn put(&mut self, m: usize, n: usize, k: usize, config: GemmConfig) {
        self.cache.insert((m, n, k), config);
    }

    /// Get or tune configuration.
    pub fn get_or_tune<F>(
        &mut self,
        a: &Matrix,
        b: &Matrix,
        tuner: &Autotuner,
        kernel_fn: F,
    ) -> Result<GemmConfig>
    where
        F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>,
    {
        let key = (a.rows, b.cols, a.cols);

        if let Some(config) = self.cache.get(&key) {
            return Ok(config.clone());
        }

        let result = tuner.tune(a, b, kernel_fn)?;
        self.cache.insert(key, result.best_config.clone());

        Ok(result.best_config)
    }

    /// Clear cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Heuristic-based configuration selector.
pub struct HeuristicSelector {
    /// Configurations for different size ranges.
    size_configs: Vec<(usize, GemmConfig)>,
}

impl Default for HeuristicSelector {
    fn default() -> Self {
        Self {
            size_configs: vec![
                (
                    64,
                    GemmConfig {
                        block_m: 32,
                        block_n: 32,
                        block_k: 8,
                        thread_m: 4,
                        thread_n: 4,
                    },
                ),
                (
                    256,
                    GemmConfig {
                        block_m: 64,
                        block_n: 64,
                        block_k: 8,
                        thread_m: 8,
                        thread_n: 8,
                    },
                ),
                (
                    1024,
                    GemmConfig {
                        block_m: 128,
                        block_n: 128,
                        block_k: 16,
                        thread_m: 8,
                        thread_n: 8,
                    },
                ),
                (
                    usize::MAX,
                    GemmConfig {
                        block_m: 256,
                        block_n: 256,
                        block_k: 16,
                        thread_m: 8,
                        thread_n: 8,
                    },
                ),
            ],
        }
    }
}

impl HeuristicSelector {
    /// Select configuration based on matrix size.
    pub fn select(&self, m: usize, n: usize, _k: usize) -> GemmConfig {
        let size = m.max(n);

        for (threshold, config) in &self.size_configs {
            if size <= *threshold {
                return config.clone();
            }
        }

        GemmConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space() {
        let space = ParameterSpace::default();
        let configs = space.generate_configs();

        // All generated configs should be valid
        for config in &configs {
            assert!(config.validate().is_ok());
        }

        assert!(!configs.is_empty());
    }

    #[test]
    fn test_heuristic_selector() {
        let selector = HeuristicSelector::default();

        let small = selector.select(32, 32, 32);
        assert_eq!(small.block_m, 32);

        let large = selector.select(2048, 2048, 2048);
        assert_eq!(large.block_m, 256);
    }

    #[test]
    fn test_tuning_cache() {
        let mut cache = TuningCache::new();
        let config = GemmConfig::default();

        cache.put(64, 64, 64, config.clone());
        assert_eq!(cache.len(), 1);

        let cached = cache.get(64, 64, 64);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().block_m, config.block_m);
    }
}
