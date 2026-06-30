"""Tests for loss functions."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_compression.losses import (
    SSIMLoss,
    MS_SSIMLoss,
    RateDistortionLoss,
    CharbonnierLoss,
)


class TestSSIMLoss:
    """Tests for SSIM loss."""

    def test_identical_images(self):
        """SSIM of identical images should be 1 (loss = 0)."""
        x = torch.rand(2, 3, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, x)
        assert loss.item() < 0.01  # Should be close to 0

    def test_different_images(self):
        """SSIM of different images should be < 1."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, y)
        assert loss.item() > 0  # Should be > 0

    def test_output_shape(self):
        """Test output is scalar."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, y)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        """Test gradients flow through."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)
        loss_fn = SSIMLoss()
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


class TestMS_SSIMLoss:
    """Tests for MS-SSIM loss."""

    def test_identical_images(self):
        """MS-SSIM of identical images."""
        x = torch.rand(2, 3, 128, 128)
        loss_fn = MS_SSIMLoss()
        loss = loss_fn(x, x)
        assert loss.item() < 0.1

    def test_different_images(self):
        """MS-SSIM of different images."""
        x = torch.rand(2, 3, 128, 128)
        y = torch.rand(2, 3, 128, 128)
        loss_fn = MS_SSIMLoss()
        loss = loss_fn(x, y)
        assert loss.item() > 0

    def test_requires_minimum_size(self):
        """MS-SSIM requires images large enough for downsampling."""
        x = torch.rand(2, 3, 128, 128)
        y = torch.rand(2, 3, 128, 128)
        loss_fn = MS_SSIMLoss()
        # Should not raise for sufficiently large images
        loss = loss_fn(x, y)
        assert loss.item() >= 0


class TestRateDistortionLoss:
    """Tests for R-D loss."""

    def test_mse_mode(self):
        """Test MSE distortion mode."""
        loss_fn = RateDistortionLoss(lambda_rd=0.01, distortion_type='mse')

        x = torch.rand(2, 3, 64, 64)
        x_hat = torch.rand(2, 3, 64, 64)
        likelihoods = {
            'y': torch.rand(2, 192, 4, 4).clamp(min=1e-9),
            'z': torch.rand(2, 128, 1, 1).clamp(min=1e-9),
        }

        loss, metrics = loss_fn(x, x_hat, likelihoods)

        assert 'loss' in metrics
        assert 'mse' in metrics
        assert 'psnr' in metrics
        assert 'bpp' in metrics

    def test_lambda_effect(self):
        """Higher lambda should increase rate penalty."""
        x = torch.rand(2, 3, 64, 64)
        x_hat = x.clone()  # Same image
        likelihoods = {
            'y': torch.rand(2, 192, 4, 4).clamp(min=1e-9),
            'z': torch.rand(2, 128, 1, 1).clamp(min=1e-9),
        }

        loss_fn_low = RateDistortionLoss(lambda_rd=0.001)
        loss_fn_high = RateDistortionLoss(lambda_rd=0.1)

        loss_low, _ = loss_fn_low(x, x_hat, likelihoods)
        loss_high, _ = loss_fn_high(x, x_hat, likelihoods)

        # Higher lambda = higher loss (rate is non-zero)
        assert loss_high > loss_low


class TestCharbonnierLoss:
    """Tests for Charbonnier loss."""

    def test_zero_difference(self):
        """Identical inputs should give near-zero loss."""
        x = torch.rand(2, 3, 32, 32)
        loss_fn = CharbonnierLoss()
        loss = loss_fn(x, x)
        assert loss.item() < 1e-3

    def test_gradient_flow(self):
        """Test gradients flow through."""
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        y = torch.rand(2, 3, 32, 32)
        loss_fn = CharbonnierLoss()
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None

    def test_smoothness(self):
        """Charbonnier should be smooth around zero."""
        x = torch.zeros(1)
        y = torch.tensor([0.001])
        loss_fn = CharbonnierLoss(epsilon=1e-3)
        loss = loss_fn(x, y)
        # Should not be exactly equal to L1 due to smoothing
        assert abs(loss.item() - 0.001) > 1e-6
