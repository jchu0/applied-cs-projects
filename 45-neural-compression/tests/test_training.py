"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_compression.training import (
    TrainingConfig,
    TrainingState,
    CompressionTrainer,
)
from neural_compression.codecs import NeuralCompressionCodec


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.lr == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 200
        assert config.lambda_rd == 0.01

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            lr=1e-3,
            batch_size=16,
            lambda_rd=0.1,
        )

        assert config.lr == 1e-3
        assert config.batch_size == 16
        assert config.lambda_rd == 0.1


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_state(self):
        """Test default state values."""
        state = TrainingState()

        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_loss == float('inf')
        assert state.best_psnr == 0.0

    def test_state_update(self):
        """Test state can be updated."""
        state = TrainingState()
        state.epoch = 10
        state.best_loss = 0.5

        assert state.epoch == 10
        assert state.best_loss == 0.5


class TestCompressionTrainer:
    """Tests for CompressionTrainer."""

    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return NeuralCompressionCodec(
            latent_channels=32,
            hyper_channels=16,
            num_filters=16,
        )

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TrainingConfig(
            lr=1e-4,
            batch_size=2,
            num_epochs=2,
            log_every=1,
            save_every=1,
        )

    def test_trainer_creation(self, model, config):
        """Test trainer can be created."""
        trainer = CompressionTrainer(model, config, device=torch.device('cpu'))

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_single_training_step(self, model, config):
        """Test single training step."""
        trainer = CompressionTrainer(model, config, device=torch.device('cpu'))

        # Create dummy batch
        batch = torch.rand(2, 3, 64, 64)

        # Create simple dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(batch)
        loader = DataLoader(dataset, batch_size=2)

        # Train one epoch
        metrics = trainer.train_epoch(loader)

        assert 'loss' in metrics
        assert 'psnr' in metrics
        assert 'bpp' in metrics

    def test_validation(self, model, config):
        """Test validation."""
        trainer = CompressionTrainer(model, config, device=torch.device('cpu'))

        batch = torch.rand(2, 3, 64, 64)

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(batch)
        loader = DataLoader(dataset, batch_size=2)

        metrics = trainer.validate(loader)

        assert 'loss' in metrics
        assert 'psnr' in metrics
        assert 'bpp' in metrics

    def test_checkpoint_save_load(self, model, config):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = CompressionTrainer(model, config, device=torch.device('cpu'))

            # Modify state
            trainer.state.epoch = 5
            trainer.state.best_loss = 0.5

            # Save
            trainer.save_checkpoint('test.pth')

            # Create new trainer and load
            trainer2 = CompressionTrainer(model, config, device=torch.device('cpu'))
            trainer2.load_checkpoint(os.path.join(tmpdir, 'test.pth'))

            assert trainer2.state.epoch == 5
            assert trainer2.state.best_loss == 0.5


class TestMultiRateTrain:
    """Tests for multi-rate training."""

    def test_import(self):
        """Test MultiRateTrain can be imported."""
        from neural_compression.training import MultiRateTrain
        assert MultiRateTrain is not None
