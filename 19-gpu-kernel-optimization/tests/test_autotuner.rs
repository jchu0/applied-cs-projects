//! Tests for the autotuner module.

use gpu_gemm_optimization::{Matrix, GemmConfig, register_tiled_gemm};
use gpu_gemm_optimization::autotuner::{
    Autotuner, AutotuneConfig, SearchStrategy, ParameterSpace, TuningCache, HeuristicSelector,
};
use std::time::Duration;

#[test]
fn test_autotuner_creation() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::GridSearch,
        max_trials: 10,
        time_budget: Some(Duration::from_secs(30)),
        early_stop: 5,
        benchmark_iters: 3,
    };

    let param_space = ParameterSpace::small();
    let tuner = Autotuner::new(config, param_space);

    assert_eq!(tuner.config.max_trials, 10);
    assert_eq!(tuner.config.early_stop, 5);
}

#[test]
fn test_autotuner_default_config() {
    let config = AutotuneConfig::default();

    assert_eq!(config.strategy, SearchStrategy::GridSearch);
    assert_eq!(config.max_trials, 100);
    assert_eq!(config.early_stop, 20);
    assert_eq!(config.benchmark_iters, 5);
}

#[test]
fn test_parameter_space_default() {
    let space = ParameterSpace::default();

    assert!(!space.block_m.is_empty());
    assert!(!space.block_n.is_empty());
    assert!(!space.block_k.is_empty());
    assert!(!space.thread_m.is_empty());
    assert!(!space.thread_n.is_empty());
}

#[test]
fn test_parameter_space_small() {
    let space = ParameterSpace::small();

    // Small space should have smaller tile sizes
    assert!(space.block_m.iter().max().unwrap() <= &64);
}

#[test]
fn test_parameter_space_large() {
    let space = ParameterSpace::large();

    // Large space should have larger tile sizes
    assert!(space.block_m.iter().max().unwrap() >= &128);
}

#[test]
fn test_parameter_space_generate_configs() {
    let space = ParameterSpace::small();
    let configs = space.generate_configs();

    // Should generate some valid configurations
    assert!(!configs.is_empty());

    // All should be valid
    for config in &configs {
        assert!(config.validate().is_ok());
    }
}

#[test]
fn test_parameter_space_num_configs() {
    let space = ParameterSpace::small();
    let count = space.num_configs();
    let configs = space.generate_configs();

    assert_eq!(count, configs.len());
}

#[test]
fn test_autotuner_grid_search() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::GridSearch,
        max_trials: 5,
        time_budget: None,
        early_stop: 10,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace {
        block_m: vec![32, 64],
        block_n: vec![32, 64],
        block_k: vec![8],
        thread_m: vec![8],
        thread_n: vec![8],
    };

    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    });

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.num_trials > 0);
    assert!(result.best_metrics.gflops > 0.0);
}

#[test]
fn test_autotuner_random_search() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::Random,
        max_trials: 5,
        time_budget: None,
        early_stop: 10,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace::small();
    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    });

    assert!(result.is_ok());
}

#[test]
fn test_autotuner_simulated_annealing() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::SimulatedAnnealing,
        max_trials: 10,
        time_budget: None,
        early_stop: 15,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace::small();
    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    });

    assert!(result.is_ok());
}

#[test]
fn test_autotuner_genetic() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::Genetic,
        max_trials: 25,
        time_budget: None,
        early_stop: 30,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace::small();
    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    });

    assert!(result.is_ok());
}

#[test]
fn test_autotune_result_format() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::GridSearch,
        max_trials: 3,
        time_budget: None,
        early_stop: 5,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace {
        block_m: vec![32],
        block_n: vec![32],
        block_k: vec![8],
        thread_m: vec![4],
        thread_n: vec![4],
    };

    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(32, 32);
    let b = Matrix::random(32, 32);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    }).unwrap();

    let formatted = result.format();
    assert!(formatted.contains("Autotuning Results"));
    assert!(formatted.contains("GFLOPS"));
}

#[test]
fn test_tuning_cache() {
    let mut cache = TuningCache::new();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    let config = GemmConfig::default();
    cache.put(64, 64, 64, config.clone());

    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);

    let cached = cache.get(64, 64, 64);
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().block_m, config.block_m);

    // Non-existent entry
    assert!(cache.get(128, 128, 128).is_none());
}

#[test]
fn test_tuning_cache_clear() {
    let mut cache = TuningCache::new();

    cache.put(64, 64, 64, GemmConfig::default());
    cache.put(128, 128, 128, GemmConfig::default());

    assert_eq!(cache.len(), 2);

    cache.clear();

    assert!(cache.is_empty());
    assert!(cache.get(64, 64, 64).is_none());
}

#[test]
fn test_heuristic_selector() {
    let selector = HeuristicSelector::default();

    // Small matrix
    let small_config = selector.select(32, 32, 32);
    assert_eq!(small_config.block_m, 32);

    // Medium matrix
    let medium_config = selector.select(128, 128, 128);
    assert!(medium_config.block_m >= 32);

    // Large matrix
    let large_config = selector.select(2048, 2048, 2048);
    assert!(large_config.block_m >= 128);
}

#[test]
fn test_autotuner_with_time_budget() {
    let config = AutotuneConfig {
        strategy: SearchStrategy::Random,
        max_trials: 1000, // High limit
        time_budget: Some(Duration::from_millis(500)), // But limited by time
        early_stop: 1000,
        benchmark_iters: 2,
    };

    let param_space = ParameterSpace::small();
    let tuner = Autotuner::new(config, param_space);

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    let result = tuner.tune(&a, &b, |a, b, c, config| {
        register_tiled_gemm(a, b, c, config)
    }).unwrap();

    // Should have stopped before max_trials due to time budget
    assert!(result.num_trials < 1000);
    assert!(result.total_time < Duration::from_secs(5));
}

#[test]
fn test_all_search_strategies() {
    let strategies = vec![
        SearchStrategy::GridSearch,
        SearchStrategy::Random,
        SearchStrategy::SimulatedAnnealing,
        SearchStrategy::Genetic,
    ];

    let param_space = ParameterSpace {
        block_m: vec![32, 64],
        block_n: vec![32, 64],
        block_k: vec![8],
        thread_m: vec![8],
        thread_n: vec![8],
    };

    let a = Matrix::random(64, 64);
    let b = Matrix::random(64, 64);

    for strategy in strategies {
        let config = AutotuneConfig {
            strategy,
            max_trials: 5,
            time_budget: None,
            early_stop: 10,
            benchmark_iters: 2,
        };

        let tuner = Autotuner::new(config, param_space.clone());

        let result = tuner.tune(&a, &b, |a, b, c, config| {
            register_tiled_gemm(a, b, c, config)
        });

        assert!(result.is_ok(), "Failed for strategy {:?}", strategy);
    }
}
