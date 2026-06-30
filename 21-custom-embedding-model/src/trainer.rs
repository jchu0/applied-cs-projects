//! Training loop for embedding models.

use crate::dataset::{EmbeddingBatch, InBatchNegativeSampler, MemoryBank};
use crate::loss::LossFunction;
use crate::model::BiEncoder;
use crate::{Error, Result, EMBEDDING_DIM};
use std::time::{Duration, Instant};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f32,
    /// Temperature for contrastive loss.
    pub temperature: f32,
    /// Use memory bank for additional negatives.
    pub use_memory_bank: bool,
    /// Memory bank size.
    pub memory_bank_size: usize,
    /// Log interval (steps).
    pub log_interval: usize,
    /// Evaluation interval (epochs).
    pub eval_interval: usize,
    /// Save checkpoint interval (epochs).
    pub save_interval: usize,
    /// Gradient accumulation steps.
    pub gradient_accumulation: usize,
    /// Warmup ratio.
    pub warmup_ratio: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            batch_size: 32,
            learning_rate: 2e-5,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            temperature: 0.05,
            use_memory_bank: true,
            memory_bank_size: 65536,
            log_interval: 100,
            eval_interval: 1,
            save_interval: 1,
            gradient_accumulation: 1,
            warmup_ratio: 0.1,
        }
    }
}

/// Training metrics.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Training loss.
    pub train_loss: f32,
    /// Number of steps.
    pub steps: usize,
    /// Training time.
    pub train_time: Duration,
    /// Samples processed.
    pub samples_processed: usize,
    /// Throughput (samples/second).
    pub throughput: f32,
}

impl TrainingMetrics {
    /// Format metrics for display.
    pub fn format(&self) -> String {
        format!(
            "Loss: {:.4} | Steps: {} | Time: {:.2}s | Throughput: {:.2} samples/s",
            self.train_loss,
            self.steps,
            self.train_time.as_secs_f32(),
            self.throughput
        )
    }
}

/// Embedding model trainer.
pub struct EmbeddingTrainer<L: LossFunction> {
    /// Model.
    model: BiEncoder,
    /// Loss function.
    loss_fn: L,
    /// Training configuration.
    config: TrainerConfig,
    /// In-batch negative sampler.
    negative_sampler: InBatchNegativeSampler,
    /// Current step.
    current_step: usize,
    /// Accumulated gradients.
    gradients: Vec<f32>,
}

impl<L: LossFunction> EmbeddingTrainer<L> {
    /// Create new trainer.
    pub fn new(model: BiEncoder, loss_fn: L, config: TrainerConfig) -> Self {
        let embedding_dim = model.output_dim();

        let negative_sampler = InBatchNegativeSampler::new(
            config.memory_bank_size,
            embedding_dim,
            config.use_memory_bank,
        );

        let num_params = model.num_parameters();
        let gradients = vec![0.0; num_params];

        Self {
            model,
            loss_fn,
            config,
            negative_sampler,
            current_step: 0,
            gradients,
        }
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        batches: &[EmbeddingBatch],
        epoch: usize,
    ) -> TrainingMetrics {
        let start = Instant::now();
        let mut total_loss = 0.0;
        let mut samples_processed = 0;

        for batch in batches {
            let batch_loss = self.train_step(batch);
            total_loss += batch_loss;
            samples_processed += batch.len();

            self.current_step += 1;

            if self.current_step % self.config.log_interval == 0 {
                let avg_loss = total_loss / (self.current_step % self.config.log_interval + 1) as f32;
                println!(
                    "Epoch {} | Step {} | Loss: {:.4}",
                    epoch, self.current_step, avg_loss
                );
            }
        }

        let train_time = start.elapsed();
        let avg_loss = total_loss / batches.len() as f32;

        TrainingMetrics {
            train_loss: avg_loss,
            steps: batches.len(),
            train_time,
            samples_processed,
            throughput: samples_processed as f32 / train_time.as_secs_f32(),
        }
    }

    /// Single training step.
    fn train_step(&mut self, batch: &EmbeddingBatch) -> f32 {
        // Get embeddings
        let anchors = &batch.anchor_embeddings;
        let positives = &batch.positive_embeddings;
        let negatives = if batch.negative_embeddings.is_empty() {
            None
        } else {
            Some(&batch.negative_embeddings)
        };

        // Get additional negatives from memory bank
        let memory_negatives = self.negative_sampler.get_negatives(positives);

        // Compute loss
        let loss = self.loss_fn.compute(anchors, positives, negatives.map(|v| &**v));

        // Update memory bank
        self.negative_sampler.update_memory(positives);

        // Simulate gradient update
        self.update_weights(loss);

        loss
    }

    /// Simulate weight update (gradient descent).
    fn update_weights(&mut self, _loss: f32) {
        // In a real implementation, this would:
        // 1. Compute gradients via backpropagation
        // 2. Clip gradients
        // 3. Apply optimizer update

        // Simulate with small random perturbations
        let lr = self.get_learning_rate();
        for param in self.model.parameters_mut() {
            for p in param.iter_mut() {
                *p -= lr * rand::random::<f32>() * 0.001;
            }
        }
    }

    /// Get current learning rate with warmup/decay.
    fn get_learning_rate(&self) -> f32 {
        let base_lr = self.config.learning_rate;
        let warmup_steps = (self.config.warmup_ratio * 1000.0) as usize;

        if self.current_step < warmup_steps {
            // Linear warmup
            base_lr * (self.current_step as f32 / warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = self.current_step as f32 / 10000.0;
            base_lr * (1.0 + progress.cos()) / 2.0
        }
    }

    /// Get model reference.
    pub fn model(&self) -> &BiEncoder {
        &self.model
    }

    /// Get mutable model reference.
    pub fn model_mut(&mut self) -> &mut BiEncoder {
        &mut self.model
    }

    /// Save checkpoint.
    pub fn save_checkpoint(&self, _path: &str) -> Result<()> {
        // In production, serialize model weights
        Ok(())
    }

    /// Load checkpoint.
    pub fn load_checkpoint(&mut self, _path: &str) -> Result<()> {
        // In production, deserialize model weights
        Ok(())
    }
}

/// Learning rate scheduler.
pub struct LRScheduler {
    /// Base learning rate.
    base_lr: f32,
    /// Warmup steps.
    warmup_steps: usize,
    /// Total steps.
    total_steps: usize,
    /// Current step.
    current_step: usize,
}

impl LRScheduler {
    /// Create cosine schedule with warmup.
    pub fn cosine_with_warmup(
        base_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Get current learning rate.
    pub fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine annealing
            let progress = (self.current_step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            self.base_lr * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
        }
    }

    /// Step scheduler.
    pub fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Optimizer state for AdamW.
pub struct AdamWOptimizer {
    /// Learning rate.
    lr: f32,
    /// Beta1 for momentum.
    beta1: f32,
    /// Beta2 for RMSProp.
    beta2: f32,
    /// Weight decay.
    weight_decay: f32,
    /// Epsilon for numerical stability.
    eps: f32,
    /// First moment estimates.
    m: Vec<f32>,
    /// Second moment estimates.
    v: Vec<f32>,
    /// Step count.
    t: usize,
}

impl AdamWOptimizer {
    /// Create new AdamW optimizer.
    pub fn new(
        num_params: usize,
        lr: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay,
            eps: 1e-8,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
        }
    }

    /// Update parameters with gradients.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Update parameters with weight decay
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * params[i]);
        }
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Gradient clipper.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let total_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    total_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::MultipleNegativesRankingLoss;
    use crate::model::random_embeddings;

    // =============================================================================
    // TrainerConfig Tests
    // =============================================================================

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();

        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.batch_size, 32);
        assert!((config.learning_rate - 2e-5).abs() < 1e-10);
        assert!((config.weight_decay - 0.01).abs() < 0.001);
        assert!((config.max_grad_norm - 1.0).abs() < 0.001);
        assert!((config.temperature - 0.05).abs() < 0.001);
        assert!(config.use_memory_bank);
        assert_eq!(config.memory_bank_size, 65536);
        assert_eq!(config.log_interval, 100);
        assert_eq!(config.eval_interval, 1);
        assert_eq!(config.save_interval, 1);
        assert_eq!(config.gradient_accumulation, 1);
        assert!((config.warmup_ratio - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_trainer_config_custom() {
        let mut config = TrainerConfig::default();
        config.num_epochs = 10;
        config.batch_size = 64;
        config.learning_rate = 1e-4;

        assert_eq!(config.num_epochs, 10);
        assert_eq!(config.batch_size, 64);
        assert!((config.learning_rate - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_config_clone() {
        let config = TrainerConfig::default();
        let cloned = config.clone();

        assert_eq!(config.num_epochs, cloned.num_epochs);
        assert_eq!(config.batch_size, cloned.batch_size);
    }

    // =============================================================================
    // TrainingMetrics Tests
    // =============================================================================

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();

        assert!((metrics.train_loss - 0.0).abs() < 0.001);
        assert_eq!(metrics.steps, 0);
        assert_eq!(metrics.samples_processed, 0);
        assert!((metrics.throughput - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_training_metrics_format() {
        let mut metrics = TrainingMetrics::default();
        metrics.train_loss = 0.5;
        metrics.steps = 100;
        metrics.samples_processed = 3200;
        metrics.train_time = std::time::Duration::from_secs(10);
        metrics.throughput = 320.0;

        let formatted = metrics.format();
        assert!(formatted.contains("Loss:"));
        assert!(formatted.contains("Steps:"));
        assert!(formatted.contains("Time:"));
        assert!(formatted.contains("Throughput:"));
    }

    // =============================================================================
    // LRScheduler Tests
    // =============================================================================

    #[test]
    fn test_lr_scheduler_creation() {
        let scheduler = LRScheduler::cosine_with_warmup(0.001, 100, 1000);
        assert!((scheduler.get_lr() - 0.0).abs() < 0.001); // Starts at 0
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let mut scheduler = LRScheduler::cosine_with_warmup(0.001, 100, 1000);

        // Initial LR should be 0
        assert!(scheduler.get_lr() < 0.0001);

        // Step through warmup
        for _ in 0..50 {
            scheduler.step();
        }
        let warmup_lr = scheduler.get_lr();
        assert!(warmup_lr > 0.0);
        assert!(warmup_lr < 0.001);
    }

    #[test]
    fn test_lr_scheduler_post_warmup() {
        let mut scheduler = LRScheduler::cosine_with_warmup(0.001, 100, 1000);

        // Complete warmup
        for _ in 0..100 {
            scheduler.step();
        }
        let lr_at_warmup_end = scheduler.get_lr();
        assert!((lr_at_warmup_end - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_lr_scheduler_cosine_decay() {
        let mut scheduler = LRScheduler::cosine_with_warmup(0.001, 100, 1000);

        // Complete warmup
        for _ in 0..100 {
            scheduler.step();
        }
        let lr_after_warmup = scheduler.get_lr();

        // Continue training
        for _ in 0..450 {
            scheduler.step();
        }
        let lr_mid = scheduler.get_lr();

        // LR should decrease with cosine decay
        assert!(lr_mid < lr_after_warmup);
        assert!(lr_mid > 0.0);
    }

    #[test]
    fn test_lr_scheduler_end_of_training() {
        let mut scheduler = LRScheduler::cosine_with_warmup(0.001, 100, 1000);

        // Run to end
        for _ in 0..1000 {
            scheduler.step();
        }
        let final_lr = scheduler.get_lr();

        // LR should be close to 0 at end
        assert!(final_lr < 0.0005);
    }

    // =============================================================================
    // AdamWOptimizer Tests
    // =============================================================================

    #[test]
    fn test_adamw_creation() {
        let optimizer = AdamWOptimizer::new(100, 0.001, 0.01);
        // Just verify creation doesn't panic
        assert!(true);
    }

    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamWOptimizer::new(4, 0.001, 0.01);
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let grads = vec![0.1, 0.2, 0.3, 0.4];

        optimizer.step(&mut params, &grads);

        // Parameters should have changed
        assert!((params[0] - 1.0).abs() > 0.0001);
    }

    #[test]
    fn test_adamw_multiple_steps() {
        let mut optimizer = AdamWOptimizer::new(4, 0.01, 0.01);
        let mut params = vec![1.0, 1.0, 1.0, 1.0];
        let grads = vec![1.0, 1.0, 1.0, 1.0];

        for _ in 0..10 {
            optimizer.step(&mut params, &grads);
        }

        // Parameters should have decreased significantly
        for p in &params {
            assert!(*p < 1.0);
        }
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut optimizer_no_decay = AdamWOptimizer::new(4, 0.01, 0.0);
        let mut optimizer_with_decay = AdamWOptimizer::new(4, 0.01, 0.1);

        let mut params_no_decay = vec![1.0, 1.0, 1.0, 1.0];
        let mut params_with_decay = vec![1.0, 1.0, 1.0, 1.0];
        let grads = vec![0.0, 0.0, 0.0, 0.0]; // No gradient

        for _ in 0..100 {
            optimizer_no_decay.step(&mut params_no_decay, &grads);
            optimizer_with_decay.step(&mut params_with_decay, &grads);
        }

        // With weight decay, parameters should be smaller
        let sum_no_decay: f32 = params_no_decay.iter().sum();
        let sum_with_decay: f32 = params_with_decay.iter().sum();
        assert!(sum_with_decay < sum_no_decay);
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut optimizer = AdamWOptimizer::new(4, 0.001, 0.01);
        optimizer.set_lr(0.01);

        let mut params = vec![1.0, 1.0, 1.0, 1.0];
        let grads = vec![1.0, 1.0, 1.0, 1.0];

        optimizer.step(&mut params, &grads);

        // With higher LR, change should be more significant
        assert!((params[0] - 1.0).abs() > 0.001);
    }

    #[test]
    fn test_adamw_momentum() {
        let mut optimizer = AdamWOptimizer::new(4, 0.01, 0.0);
        let mut params = vec![0.0, 0.0, 0.0, 0.0];
        let grads = vec![1.0, 1.0, 1.0, 1.0];

        // First step
        optimizer.step(&mut params, &grads);
        let change1 = params[0].abs();

        // Second step with same gradient - momentum should accelerate
        optimizer.step(&mut params, &grads);
        let change2 = (params[0] - (-change1)).abs();

        // Due to momentum, second update should be larger
        // (accounting for bias correction in early steps)
        assert!(change2 > 0.0);
    }

    // =============================================================================
    // Gradient Clipping Tests
    // =============================================================================

    #[test]
    fn test_clip_grad_norm_basic() {
        let mut grads = vec![3.0, 4.0]; // norm = 5
        let norm = clip_grad_norm(&mut grads, 1.0);

        assert!((norm - 5.0).abs() < 0.001);

        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_grad_norm_no_clip_needed() {
        let mut grads = vec![0.3, 0.4]; // norm = 0.5
        let norm = clip_grad_norm(&mut grads, 1.0);

        assert!((norm - 0.5).abs() < 0.001);

        // Gradients should remain unchanged
        assert!((grads[0] - 0.3).abs() < 0.001);
        assert!((grads[1] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_clip_grad_norm_large_gradients() {
        let mut grads = vec![30.0, 40.0]; // norm = 50
        let norm = clip_grad_norm(&mut grads, 5.0);

        assert!((norm - 50.0).abs() < 0.001);

        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_grad_norm_high_dimensional() {
        let mut grads: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let original_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();

        let norm = clip_grad_norm(&mut grads, 1.0);

        assert!((norm - original_norm).abs() < 0.001);

        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        if original_norm > 1.0 {
            assert!((new_norm - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_clip_grad_norm_zero_gradients() {
        let mut grads = vec![0.0, 0.0, 0.0];
        let norm = clip_grad_norm(&mut grads, 1.0);

        assert!((norm - 0.0).abs() < 0.001);
        assert_eq!(grads, vec![0.0, 0.0, 0.0]);
    }

    // =============================================================================
    // EmbeddingTrainer Tests
    // =============================================================================

    #[test]
    fn test_trainer_creation() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let trainer = EmbeddingTrainer::new(model, loss_fn, config);
        assert!(trainer.model().embedding_dim == 64);
    }

    #[test]
    fn test_trainer_train_epoch() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        // Create test batch
        let mut batch = EmbeddingBatch::new(4);
        for _ in 0..4 {
            let anchor = random_embeddings(1, 64, true).pop().unwrap();
            let positive = anchor.clone();
            batch.add(anchor, positive, vec![]);
        }

        let metrics = trainer.train_epoch(&[batch], 0);
        assert!(metrics.train_loss >= 0.0);
        assert!(metrics.steps > 0);
        assert!(metrics.samples_processed > 0);
    }

    #[test]
    fn test_trainer_multiple_batches() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        // Create multiple batches
        let mut batches = Vec::new();
        for _ in 0..5 {
            let mut batch = EmbeddingBatch::new(8);
            for _ in 0..8 {
                let anchor = random_embeddings(1, 64, true).pop().unwrap();
                let positive = random_embeddings(1, 64, true).pop().unwrap();
                batch.add(anchor, positive, vec![]);
            }
            batches.push(batch);
        }

        let metrics = trainer.train_epoch(&batches, 0);
        assert_eq!(metrics.steps, 5);
        assert_eq!(metrics.samples_processed, 40);
    }

    #[test]
    fn test_trainer_with_negatives() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        let mut batch = EmbeddingBatch::new(4);
        for _ in 0..4 {
            let anchor = random_embeddings(1, 64, true).pop().unwrap();
            let positive = random_embeddings(1, 64, true).pop().unwrap();
            let negatives = random_embeddings(3, 64, true);
            batch.add(anchor, positive, negatives);
        }

        let metrics = trainer.train_epoch(&[batch], 0);
        assert!(metrics.train_loss >= 0.0);
    }

    #[test]
    fn test_trainer_multiple_epochs() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        let mut batch = EmbeddingBatch::new(4);
        for _ in 0..4 {
            let anchor = random_embeddings(1, 64, true).pop().unwrap();
            let positive = anchor.clone();
            batch.add(anchor, positive, vec![]);
        }

        let batches = vec![batch.clone(), batch.clone()];

        // Train for multiple epochs
        for epoch in 0..3 {
            let metrics = trainer.train_epoch(&batches, epoch);
            assert!(metrics.train_loss >= 0.0);
        }
    }

    #[test]
    fn test_trainer_model_access() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let trainer = EmbeddingTrainer::new(model, loss_fn, config);

        assert_eq!(trainer.model().embedding_dim, 64);
        assert_eq!(trainer.model().output_dim(), 64);
    }

    #[test]
    fn test_trainer_model_mut() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        // Access mutable model
        let _model_mut = trainer.model_mut();
        assert!(true);
    }

    #[test]
    fn test_trainer_save_load_checkpoint() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        // Save checkpoint (currently a no-op)
        let save_result = trainer.save_checkpoint("/tmp/checkpoint");
        assert!(save_result.is_ok());

        // Load checkpoint (currently a no-op)
        let load_result = trainer.load_checkpoint("/tmp/checkpoint");
        assert!(load_result.is_ok());
    }

    #[test]
    fn test_trainer_empty_batches() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        let batches: Vec<EmbeddingBatch> = vec![];
        let metrics = trainer.train_epoch(&batches, 0);

        assert_eq!(metrics.steps, 0);
    }

    // =============================================================================
    // Integration Tests
    // =============================================================================

    #[test]
    fn test_full_training_loop() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let mut config = TrainerConfig::default();
        config.num_epochs = 2;
        config.log_interval = 1000; // Disable logging for test

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config.clone());

        // Create training data
        let mut batches = Vec::new();
        for _ in 0..3 {
            let mut batch = EmbeddingBatch::new(8);
            for _ in 0..8 {
                let anchor = random_embeddings(1, 64, true).pop().unwrap();
                let positive = random_embeddings(1, 64, true).pop().unwrap();
                batch.add(anchor, positive, vec![]);
            }
            batches.push(batch);
        }

        // Train for multiple epochs
        let mut all_metrics = Vec::new();
        for epoch in 0..config.num_epochs {
            let metrics = trainer.train_epoch(&batches, epoch);
            all_metrics.push(metrics);
        }

        // Verify training ran
        assert_eq!(all_metrics.len(), 2);
        for metrics in all_metrics {
            assert!(metrics.train_loss >= 0.0);
            assert!(metrics.steps > 0);
        }
    }

    #[test]
    fn test_trainer_throughput() {
        let model = BiEncoder::new(64, crate::model::PoolingStrategy::Mean, true, None);
        let loss_fn = MultipleNegativesRankingLoss::default();
        let config = TrainerConfig::default();

        let mut trainer = EmbeddingTrainer::new(model, loss_fn, config);

        let mut batch = EmbeddingBatch::new(32);
        for _ in 0..32 {
            let anchor = random_embeddings(1, 64, true).pop().unwrap();
            let positive = anchor.clone();
            batch.add(anchor, positive, vec![]);
        }

        let metrics = trainer.train_epoch(&[batch], 0);

        // Verify throughput is calculated
        assert!(metrics.throughput > 0.0);
        assert!(metrics.train_time.as_secs_f32() > 0.0);
    }
}
