"""Training utilities for neural compression models.

This module implements:
- CompressionTrainer: Full training loop
- LearningRateScheduler: Custom LR scheduling
- Checkpoint management
"""

import os
import time
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .losses import RateDistortionLoss


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Learning rate
    lr: float = 1e-4
    lr_decay: float = 0.1
    lr_decay_steps: List[int] = field(default_factory=lambda: [100, 150])

    # Optimization
    batch_size: int = 8
    num_epochs: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.0

    # Rate-distortion
    lambda_rd: float = 0.01
    distortion_type: str = "mse"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100

    # Aux loss for entropy model parameters
    aux_loss_weight: float = 1.0


@dataclass
class TrainingState:
    """State for resumable training."""

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_psnr: float = 0.0


class CompressionTrainer:
    """Trainer for neural compression models.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        model: Neural compression model
        config: Training configuration
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        # Optimizer
        main_params = []
        aux_params = []
        for name, param in self.model.named_parameters():
            if "entropy" in name.lower() or "prior" in name.lower():
                aux_params.append(param)
            else:
                main_params.append(param)

        self.optimizer = optim.Adam(
            [
                {"params": main_params, "lr": config.lr},
                {"params": aux_params, "lr": config.lr * 10},  # Higher LR for entropy
            ],
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_decay_steps,
            gamma=config.lr_decay,
        )

        # Loss function
        self.criterion = RateDistortionLoss(
            lambda_rd=config.lambda_rd,
            distortion_type=config.distortion_type,
        )

        # Training state
        self.state = TrainingState()

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_psnr": [],
            "train_bpp": [],
            "val_loss": [],
            "val_psnr": [],
            "val_bpp": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of average metrics
        """
        self.model.train()

        epoch_metrics = {
            "loss": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "bpp": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            # Forward pass
            x_hat, losses = self.model(x)

            # Compute loss
            loss, metrics = self.criterion(x, x_hat, losses)

            # Add auxiliary loss for entropy model
            if hasattr(self.model, "entropy_model"):
                aux_loss = self._compute_aux_loss()
                loss = loss + self.config.aux_loss_weight * aux_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key].item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(dataloader)}: "
                    f"Loss={metrics['loss'].item():.4f}, "
                    f"PSNR={metrics['psnr'].item():.2f}dB, "
                    f"BPP={metrics['bpp'].item():.4f}"
                )

            self.state.global_step += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of average metrics
        """
        self.model.eval()

        val_metrics = {
            "loss": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "bpp": 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            # Forward pass
            x_hat, losses = self.model(x)

            # Compute loss
            _, metrics = self.criterion(x, x_hat, losses)

            # Accumulate metrics
            for key in val_metrics:
                if key in metrics:
                    val_metrics[key] += metrics[key].item()
            num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)

        return val_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            callbacks: Optional list of callback functions

        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Lambda: {self.config.lambda_rd}")
        print(f"Epochs: {self.config.num_epochs}")

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_psnr"].append(train_metrics["psnr"])
            self.history["train_bpp"].append(train_metrics["bpp"])

            print(
                f"Train: Loss={train_metrics['loss']:.4f}, "
                f"PSNR={train_metrics['psnr']:.2f}dB, "
                f"BPP={train_metrics['bpp']:.4f}"
            )

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_psnr"].append(val_metrics["psnr"])
                self.history["val_bpp"].append(val_metrics["bpp"])

                print(
                    f"Val:   Loss={val_metrics['loss']:.4f}, "
                    f"PSNR={val_metrics['psnr']:.2f}dB, "
                    f"BPP={val_metrics['bpp']:.4f}"
                )

                # Track best
                if val_metrics["loss"] < self.state.best_loss:
                    self.state.best_loss = val_metrics["loss"]
                    self.state.best_psnr = val_metrics["psnr"]
                    self.save_checkpoint("best.pth")

            # LR scheduler
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_metrics, val_metrics if val_loader else None)

            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.1f}s")

        # Final checkpoint
        self.save_checkpoint("final.pth")

        return self.history

    def _compute_aux_loss(self) -> torch.Tensor:
        """Compute auxiliary loss for entropy model parameters."""
        aux_loss = torch.tensor(0.0, device=self.device)

        # Get entropy model parameters that need regularization
        if hasattr(self.model, "entropy_model"):
            entropy_model = self.model.entropy_model
            if hasattr(entropy_model, "hyper_entropy"):
                # Regularize factorized prior parameters
                prior = entropy_model.hyper_entropy
                if hasattr(prior, "log_scale"):
                    aux_loss = aux_loss + prior.log_scale.abs().mean()

        return aux_loss

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": self.state.__dict__,
            "config": self.config.__dict__,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        for key, value in checkpoint["state"].items():
            setattr(self.state, key, value)

        self.history = checkpoint.get("history", self.history)

        print(f"Loaded checkpoint: {path}")
        print(f"Resuming from epoch {self.state.epoch}")


class MultiRateTrain:
    """Training for multi-rate compression models.

    Trains a single model to support multiple rate points.

    Args:
        model: Multi-rate codec model
        lambdas: List of lambda values for different rates
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        lambdas: List[float],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.lambdas = lambdas
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train_step(
        self, x: torch.Tensor, rate_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Single training step with random or specified rate.

        Args:
            x: Input batch
            rate_idx: Optional rate index (random if None)

        Returns:
            Metrics dictionary
        """
        if rate_idx is None:
            rate_idx = torch.randint(0, len(self.lambdas), (1,)).item()

        x = x.to(self.device)
        lambda_rd = self.lambdas[rate_idx]

        # Forward pass
        if hasattr(self.model, "forward_rate"):
            x_hat, losses = self.model.forward_rate(x, rate_idx)
        else:
            x_hat, losses = self.model(x)

        # Compute loss
        mse = nn.functional.mse_loss(x_hat, x)
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp = (
            -torch.log2(losses["y"].clamp(min=1e-9)).sum()
            - torch.log2(losses["z"].clamp(min=1e-9)).sum()
        ) / num_pixels

        loss = mse + lambda_rd * bpp

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "mse": mse,
            "bpp": bpp,
            "rate_idx": rate_idx,
            "lambda": lambda_rd,
        }
