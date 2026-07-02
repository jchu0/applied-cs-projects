//! MLOps integration for experiment tracking, model versioning, and deployment.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for experiment tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment name.
    pub name: String,
    /// Project name.
    pub project: String,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Description.
    pub description: Option<String>,
    /// Storage path for artifacts.
    pub artifact_path: PathBuf,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "embedding-experiment".to_string(),
            project: "custom-embedding-model".to_string(),
            tags: vec![],
            description: None,
            artifact_path: PathBuf::from("./mlruns"),
        }
    }
}

/// Run status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is in progress.
    Running,
    /// Run completed successfully.
    Finished,
    /// Run failed.
    Failed,
    /// Run was killed.
    Killed,
}

/// Metric data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric value.
    pub value: f64,
    /// Step (e.g., epoch or iteration).
    pub step: u64,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Experiment run representing a single training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Unique run ID.
    pub run_id: uuid::Uuid,
    /// Experiment ID.
    pub experiment_id: String,
    /// Run name.
    pub name: String,
    /// Status.
    pub status: RunStatus,
    /// Start time.
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time.
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Parameters.
    pub params: HashMap<String, String>,
    /// Metrics (name -> history).
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    /// Tags.
    pub tags: HashMap<String, String>,
    /// Artifact paths.
    pub artifacts: Vec<PathBuf>,
}

impl ExperimentRun {
    /// Create a new experiment run.
    pub fn new(experiment_id: &str, name: &str) -> Self {
        Self {
            run_id: uuid::Uuid::new_v4(),
            experiment_id: experiment_id.to_string(),
            name: name.to_string(),
            status: RunStatus::Running,
            start_time: chrono::Utc::now(),
            end_time: None,
            params: HashMap::new(),
            metrics: HashMap::new(),
            tags: HashMap::new(),
            artifacts: Vec::new(),
        }
    }

    /// Log a parameter.
    pub fn log_param(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }

    /// Log multiple parameters.
    pub fn log_params(&mut self, params: HashMap<String, String>) {
        self.params.extend(params);
    }

    /// Log a metric value.
    pub fn log_metric(&mut self, name: &str, value: f64, step: u64) {
        let point = MetricPoint {
            value,
            step,
            timestamp: chrono::Utc::now(),
        };
        self.metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(point);
    }

    /// Log multiple metrics at once.
    pub fn log_metrics(&mut self, metrics: HashMap<String, f64>, step: u64) {
        for (name, value) in metrics {
            self.log_metric(&name, value, step);
        }
    }

    /// Set a tag.
    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    /// Log an artifact path.
    pub fn log_artifact(&mut self, path: PathBuf) {
        self.artifacts.push(path);
    }

    /// Get the latest value of a metric.
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).and_then(|h| h.last().map(|p| p.value))
    }

    /// Finish the run.
    pub fn finish(&mut self, status: RunStatus) {
        self.status = status;
        self.end_time = Some(chrono::Utc::now());
    }

    /// Get run duration in seconds.
    pub fn duration_seconds(&self) -> Option<i64> {
        self.end_time
            .map(|end| (end - self.start_time).num_seconds())
    }
}

/// Experiment tracker for managing training experiments.
pub struct ExperimentTracker {
    /// Configuration.
    config: ExperimentConfig,
    /// Experiment ID.
    experiment_id: String,
    /// All runs.
    runs: HashMap<uuid::Uuid, ExperimentRun>,
    /// Active run.
    active_run: Option<uuid::Uuid>,
}

impl ExperimentTracker {
    /// Create a new experiment tracker.
    pub fn new(config: ExperimentConfig) -> Self {
        let experiment_id = format!("{}_{}", config.project, config.name);
        Self {
            config,
            experiment_id,
            runs: HashMap::new(),
            active_run: None,
        }
    }

    /// Start a new run.
    pub fn start_run(&mut self, name: &str) -> uuid::Uuid {
        let run = ExperimentRun::new(&self.experiment_id, name);
        let run_id = run.run_id;
        self.runs.insert(run_id, run);
        self.active_run = Some(run_id);
        run_id
    }

    /// Get the active run.
    pub fn active_run(&self) -> Option<&ExperimentRun> {
        self.active_run.and_then(|id| self.runs.get(&id))
    }

    /// Get the active run mutably.
    pub fn active_run_mut(&mut self) -> Option<&mut ExperimentRun> {
        self.active_run.and_then(|id| self.runs.get_mut(&id))
    }

    /// Log a parameter to the active run.
    pub fn log_param(&mut self, key: &str, value: &str) -> Result<()> {
        self.active_run_mut()
            .ok_or(Error::InvalidConfig("No active run".to_string()))?
            .log_param(key, value);
        Ok(())
    }

    /// Log a metric to the active run.
    pub fn log_metric(&mut self, name: &str, value: f64, step: u64) -> Result<()> {
        self.active_run_mut()
            .ok_or(Error::InvalidConfig("No active run".to_string()))?
            .log_metric(name, value, step);
        Ok(())
    }

    /// End the active run.
    pub fn end_run(&mut self, status: RunStatus) -> Result<()> {
        self.active_run_mut()
            .ok_or(Error::InvalidConfig("No active run".to_string()))?
            .finish(status);
        self.active_run = None;
        Ok(())
    }

    /// Get a run by ID.
    pub fn get_run(&self, run_id: &uuid::Uuid) -> Option<&ExperimentRun> {
        self.runs.get(run_id)
    }

    /// Get all runs.
    pub fn list_runs(&self) -> Vec<&ExperimentRun> {
        self.runs.values().collect()
    }

    /// Save experiment data to JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let data = serde_json::json!({
            "experiment_id": self.experiment_id,
            "config": self.config,
            "runs": self.runs,
        });
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &data)
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    /// Load experiment data from JSON.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let data: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;

        let config: ExperimentConfig = serde_json::from_value(data["config"].clone())
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;
        let runs: HashMap<uuid::Uuid, ExperimentRun> =
            serde_json::from_value(data["runs"].clone())
                .map_err(|e| Error::InvalidConfig(e.to_string()))?;

        Ok(Self {
            experiment_id: data["experiment_id"].as_str().unwrap_or_default().to_string(),
            config,
            runs,
            active_run: None,
        })
    }
}

/// Model registry for versioning and deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name.
    pub name: String,
    /// Version number.
    pub version: u32,
    /// Stage (staging, production, archived).
    pub stage: ModelStage,
    /// Run ID that produced this model.
    pub run_id: uuid::Uuid,
    /// Model path.
    pub model_path: PathBuf,
    /// Creation time.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Description.
    pub description: Option<String>,
    /// Metrics at time of registration.
    pub metrics: HashMap<String, f64>,
}

/// Model stage for deployment lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStage {
    /// Model is in development/testing.
    None,
    /// Model is in staging environment.
    Staging,
    /// Model is in production.
    Production,
    /// Model has been archived.
    Archived,
}

/// Model registry for managing model versions.
pub struct ModelRegistry {
    /// Registered models.
    models: HashMap<String, Vec<ModelVersion>>,
    /// Storage path.
    storage_path: PathBuf,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new(storage_path: PathBuf) -> Self {
        Self {
            models: HashMap::new(),
            storage_path,
        }
    }

    /// Register a new model version.
    pub fn register_model(
        &mut self,
        name: &str,
        run_id: uuid::Uuid,
        model_path: PathBuf,
        metrics: HashMap<String, f64>,
    ) -> ModelVersion {
        let versions = self.models.entry(name.to_string()).or_insert_with(Vec::new);
        let version = versions.len() as u32 + 1;

        let model = ModelVersion {
            name: name.to_string(),
            version,
            stage: ModelStage::None,
            run_id,
            model_path,
            created_at: chrono::Utc::now(),
            description: None,
            metrics,
        };

        versions.push(model.clone());
        model
    }

    /// Transition model to a new stage.
    pub fn transition_stage(
        &mut self,
        name: &str,
        version: u32,
        stage: ModelStage,
    ) -> Result<()> {
        let versions = self
            .models
            .get_mut(name)
            .ok_or(Error::InvalidConfig("Model not found".to_string()))?;

        // Check if version exists
        let version_exists = versions.iter().any(|m| m.version == version);
        if !version_exists {
            return Err(Error::InvalidConfig("Version not found".to_string()));
        }

        // If transitioning to Production, demote current production model
        if stage == ModelStage::Production {
            for m in versions.iter_mut() {
                if m.stage == ModelStage::Production {
                    m.stage = ModelStage::Archived;
                }
            }
        }

        // Now update the target version
        if let Some(model) = versions.iter_mut().find(|m| m.version == version) {
            model.stage = stage;
        }
        Ok(())
    }

    /// Get the production model.
    pub fn get_production_model(&self, name: &str) -> Option<&ModelVersion> {
        self.models.get(name).and_then(|versions| {
            versions.iter().find(|m| m.stage == ModelStage::Production)
        })
    }

    /// Get the latest model version.
    pub fn get_latest_version(&self, name: &str) -> Option<&ModelVersion> {
        self.models.get(name).and_then(|v| v.last())
    }

    /// List all versions of a model.
    pub fn list_versions(&self, name: &str) -> Vec<&ModelVersion> {
        self.models
            .get(name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// List all model names.
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Delete a model version.
    pub fn delete_version(&mut self, name: &str, version: u32) -> Result<()> {
        let versions = self
            .models
            .get_mut(name)
            .ok_or(Error::InvalidConfig("Model not found".to_string()))?;

        versions.retain(|m| m.version != version);
        Ok(())
    }

    /// Save registry to JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &self.models)
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    /// Load registry from JSON.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let models: HashMap<String, Vec<ModelVersion>> = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;

        Ok(Self {
            models,
            storage_path: path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf(),
        })
    }
}

/// Hyperparameter search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Parameter name.
    pub name: String,
    /// Search type.
    pub search_type: SearchType,
}

/// Hyperparameter search type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    /// Categorical choice.
    Choice(Vec<serde_json::Value>),
    /// Uniform range (min, max).
    Uniform(f64, f64),
    /// Log-uniform range (min, max).
    LogUniform(f64, f64),
    /// Integer range (min, max).
    IntRange(i64, i64),
}

impl SearchType {
    /// Sample a value from the search space.
    pub fn sample(&self) -> serde_json::Value {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        match self {
            SearchType::Choice(choices) => {
                let idx = rng.gen_range(0..choices.len());
                choices[idx].clone()
            }
            SearchType::Uniform(min, max) => {
                serde_json::json!(rng.gen_range(*min..*max))
            }
            SearchType::LogUniform(min, max) => {
                let log_val = rng.gen_range(min.ln()..max.ln());
                serde_json::json!(log_val.exp())
            }
            SearchType::IntRange(min, max) => {
                serde_json::json!(rng.gen_range(*min..=*max))
            }
        }
    }
}

/// Hyperparameter search runner.
pub struct HyperparameterSearch {
    /// Search space.
    space: Vec<HyperparameterSpace>,
    /// Number of trials.
    n_trials: usize,
    /// Best parameters found.
    best_params: Option<HashMap<String, serde_json::Value>>,
    /// Best score found.
    best_score: Option<f64>,
}

impl HyperparameterSearch {
    /// Create a new hyperparameter search.
    pub fn new(space: Vec<HyperparameterSpace>, n_trials: usize) -> Self {
        Self {
            space,
            n_trials,
            best_params: None,
            best_score: None,
        }
    }

    /// Generate a random configuration.
    pub fn sample_config(&self) -> HashMap<String, serde_json::Value> {
        self.space
            .iter()
            .map(|p| (p.name.clone(), p.search_type.sample()))
            .collect()
    }

    /// Record a trial result.
    pub fn record_trial(&mut self, params: HashMap<String, serde_json::Value>, score: f64) {
        if self.best_score.map_or(true, |best| score > best) {
            self.best_score = Some(score);
            self.best_params = Some(params);
        }
    }

    /// Get best parameters.
    pub fn best_params(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.best_params.as_ref()
    }

    /// Get best score.
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Get number of trials.
    pub fn n_trials(&self) -> usize {
        self.n_trials
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_run_basic() {
        let mut run = ExperimentRun::new("exp-1", "run-1");
        assert_eq!(run.status, RunStatus::Running);
        assert!(run.end_time.is_none());

        run.log_param("learning_rate", "0.001");
        run.log_metric("loss", 0.5, 0);
        run.log_metric("loss", 0.3, 1);

        assert_eq!(run.params.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(run.get_metric("loss"), Some(0.3));
    }

    #[test]
    fn test_experiment_run_finish() {
        let mut run = ExperimentRun::new("exp-1", "run-1");
        run.finish(RunStatus::Finished);

        assert_eq!(run.status, RunStatus::Finished);
        assert!(run.end_time.is_some());
        assert!(run.duration_seconds().is_some());
    }

    #[test]
    fn test_experiment_tracker_basic() {
        let config = ExperimentConfig::default();
        let mut tracker = ExperimentTracker::new(config);

        let run_id = tracker.start_run("test-run");
        assert!(tracker.active_run().is_some());

        tracker.log_param("lr", "0.001").unwrap();
        tracker.log_metric("loss", 0.5, 0).unwrap();
        tracker.end_run(RunStatus::Finished).unwrap();

        assert!(tracker.active_run().is_none());
        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.status, RunStatus::Finished);
    }

    #[test]
    fn test_model_registry_basic() {
        let mut registry = ModelRegistry::new(PathBuf::from("./models"));

        let model = registry.register_model(
            "embedding-model",
            uuid::Uuid::new_v4(),
            PathBuf::from("./models/v1"),
            HashMap::from([("recall@10".to_string(), 0.85)]),
        );

        assert_eq!(model.version, 1);
        assert_eq!(model.stage, ModelStage::None);
    }

    #[test]
    fn test_model_registry_stage_transition() {
        let mut registry = ModelRegistry::new(PathBuf::from("./models"));

        registry.register_model(
            "model",
            uuid::Uuid::new_v4(),
            PathBuf::from("./v1"),
            HashMap::new(),
        );

        registry.transition_stage("model", 1, ModelStage::Production).unwrap();

        let prod = registry.get_production_model("model").unwrap();
        assert_eq!(prod.version, 1);
        assert_eq!(prod.stage, ModelStage::Production);
    }

    #[test]
    fn test_model_registry_auto_archive() {
        let mut registry = ModelRegistry::new(PathBuf::from("./models"));

        registry.register_model(
            "model",
            uuid::Uuid::new_v4(),
            PathBuf::from("./v1"),
            HashMap::new(),
        );
        registry.register_model(
            "model",
            uuid::Uuid::new_v4(),
            PathBuf::from("./v2"),
            HashMap::new(),
        );

        registry.transition_stage("model", 1, ModelStage::Production).unwrap();
        registry.transition_stage("model", 2, ModelStage::Production).unwrap();

        // v1 should be archived
        let versions = registry.list_versions("model");
        let v1 = versions.iter().find(|m| m.version == 1).unwrap();
        assert_eq!(v1.stage, ModelStage::Archived);
    }

    #[test]
    fn test_hyperparameter_search_sampling() {
        let space = vec![
            HyperparameterSpace {
                name: "lr".to_string(),
                search_type: SearchType::LogUniform(1e-5, 1e-2),
            },
            HyperparameterSpace {
                name: "batch_size".to_string(),
                search_type: SearchType::Choice(vec![
                    serde_json::json!(16),
                    serde_json::json!(32),
                    serde_json::json!(64),
                ]),
            },
        ];

        let search = HyperparameterSearch::new(space, 10);
        let config = search.sample_config();

        assert!(config.contains_key("lr"));
        assert!(config.contains_key("batch_size"));
    }

    #[test]
    fn test_hyperparameter_search_best_tracking() {
        let space = vec![HyperparameterSpace {
            name: "x".to_string(),
            search_type: SearchType::Uniform(0.0, 1.0),
        }];

        let mut search = HyperparameterSearch::new(space, 5);

        search.record_trial(HashMap::from([("x".to_string(), serde_json::json!(0.3))]), 0.7);
        search.record_trial(HashMap::from([("x".to_string(), serde_json::json!(0.5))]), 0.9);
        search.record_trial(HashMap::from([("x".to_string(), serde_json::json!(0.7))]), 0.8);

        assert_eq!(search.best_score(), Some(0.9));
    }

    #[test]
    fn test_search_type_int_range() {
        let search = SearchType::IntRange(1, 10);
        for _ in 0..100 {
            let val = search.sample();
            let v = val.as_i64().unwrap();
            assert!(v >= 1 && v <= 10);
        }
    }

    #[test]
    fn test_search_type_uniform() {
        let search = SearchType::Uniform(0.0, 1.0);
        for _ in 0..100 {
            let val = search.sample();
            let v = val.as_f64().unwrap();
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_metric_point_timestamp() {
        let point = MetricPoint {
            value: 0.5,
            step: 1,
            timestamp: chrono::Utc::now(),
        };
        assert!(point.timestamp <= chrono::Utc::now());
    }

    #[test]
    fn test_record_trial_tolerates_nan_score() {
        // record_trial must not panic on a NaN score (previously relied on an
        // is_none()/unwrap() pattern). A NaN never beats an existing best.
        let mut search = HyperparameterSearch::new(vec![], 3);
        search.record_trial(HashMap::new(), 0.8);
        search.record_trial(HashMap::new(), f64::NAN);
        assert_eq!(search.best_score(), Some(0.8));
    }
}
