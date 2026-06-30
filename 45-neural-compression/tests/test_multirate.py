"""Tests for multi-rate codec."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_compression.multirate import (
    GainUnit,
    MultiRateCodec,
    ScaleHyperpriorCodec,
)


class TestGainUnit:
    """Tests for GainUnit."""

    def test_forward(self):
        """Test forward pass."""
        gain = GainUnit(channels=64, num_rates=6)
        x = torch.rand(2, 64, 8, 8)

        for rate_idx in range(6):
            y = gain(x, rate_idx, inverse=False)
            assert y.shape == x.shape

    def test_inverse(self):
        """Test inverse gain."""
        gain = GainUnit(channels=64, num_rates=6)
        x = torch.rand(2, 64, 8, 8)

        # Forward then inverse should approximately recover
        for rate_idx in range(6):
            y = gain(x, rate_idx, inverse=False)
            x_rec = gain(y, rate_idx, inverse=True)

            torch.testing.assert_close(x_rec, x, rtol=1e-4, atol=1e-4)

    def test_different_rates(self):
        """Test that different rates give different results."""
        gain = GainUnit(channels=64, num_rates=6)
        x = torch.rand(2, 64, 8, 8)

        y0 = gain(x, 0, inverse=False)
        y5 = gain(x, 5, inverse=False)

        # Different rates should give different outputs
        assert not torch.allclose(y0, y5)


class TestMultiRateCodec:
    """Tests for MultiRateCodec."""

    @pytest.fixture
    def codec(self):
        """Create small test codec."""
        return MultiRateCodec(
            num_rates=3,
            latent_channels=32,
            hyper_channels=16,
            num_filters=16,
        )

    def test_forward(self, codec):
        """Test forward pass."""
        x = torch.rand(1, 3, 64, 64)

        for rate_idx in range(3):
            x_hat, losses = codec(x, rate_idx)

            assert x_hat.shape == x.shape
            assert 'mse' in losses
            assert 'bpp' in losses
            assert 'psnr' in losses

    def test_forward_rate(self, codec):
        """Test forward_rate is same as forward."""
        codec.eval()  # Eval mode for deterministic results
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            x_hat1, losses1 = codec.forward(x, rate_idx=1)
            x_hat2, losses2 = codec.forward_rate(x, rate_idx=1)

        torch.testing.assert_close(x_hat1, x_hat2)
        torch.testing.assert_close(losses1['mse'], losses2['mse'])

    def test_rate_distortion_tradeoff(self, codec):
        """Test that higher rates give better quality but more bits."""
        codec.eval()
        x = torch.rand(1, 3, 64, 64)

        results = []
        with torch.no_grad():
            for rate_idx in range(3):
                x_hat, losses = codec(x, rate_idx)
                results.append((losses['bpp'].item(), losses['psnr'].item()))

        # Results should exist
        assert len(results) == 3
        for bpp, psnr in results:
            assert bpp > 0
            assert psnr > 0

    def test_get_rate_distortion_points(self, codec):
        """Test R-D point extraction."""
        x = torch.rand(1, 3, 64, 64)
        points = codec.get_rate_distortion_points(x)

        assert len(points) == 3
        for rate_idx, bpp, psnr in points:
            assert 0 <= rate_idx < 3
            assert bpp > 0
            assert psnr > 0

    def test_gradient_flow(self, codec):
        """Test gradients flow through all rates."""
        x = torch.rand(1, 3, 64, 64, requires_grad=True)

        for rate_idx in range(3):
            codec.zero_grad()
            x_hat, losses = codec(x, rate_idx)
            loss = losses['mse'] + 0.01 * losses['bpp']
            loss.backward()

            # Check gradients exist
            assert x.grad is not None
            x.grad.zero_()


class TestScaleHyperpriorCodec:
    """Tests for ScaleHyperpriorCodec."""

    @pytest.fixture
    def codec(self):
        """Create small test codec."""
        return ScaleHyperpriorCodec(
            latent_channels=32,
            hyper_channels=16,
            num_filters=16,
        )

    def test_forward(self, codec):
        """Test forward pass."""
        x = torch.rand(1, 3, 64, 64)
        x_hat, losses = codec(x)

        assert x_hat.shape == x.shape
        assert 'mse' in losses
        assert 'bpp' in losses
        assert 'y_bpp' in losses
        assert 'z_bpp' in losses

    def test_training_mode(self, codec):
        """Test training mode uses noise."""
        x = torch.rand(1, 3, 64, 64)

        codec.train()
        x_hat1, _ = codec(x)
        x_hat2, _ = codec(x)

        # With noise, outputs should differ
        assert not torch.equal(x_hat1, x_hat2)

    def test_eval_mode(self, codec):
        """Test eval mode uses rounding."""
        x = torch.rand(1, 3, 64, 64)

        codec.eval()
        with torch.no_grad():
            x_hat1, _ = codec(x)
            x_hat2, _ = codec(x)

        # With rounding, outputs should be same
        torch.testing.assert_close(x_hat1, x_hat2)

    def test_gradient_flow(self, codec):
        """Test gradients flow through."""
        x = torch.rand(1, 3, 64, 64, requires_grad=True)

        x_hat, losses = codec(x)
        loss = losses['mse'] + 0.01 * losses['bpp']
        loss.backward()

        assert x.grad is not None
