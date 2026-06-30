"""Multi-rate compression codec.

This module implements:
- MultiRateCodec: Single model supporting multiple rate points
- GainUnit: Learnable gain for rate control
"""

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import AnalysisTransform, SynthesisTransform
from .entropy import EntropyModel


class GainUnit(nn.Module):
    """Learnable gain unit for rate control.

    Scales latent representations to achieve different rate points.

    Args:
        channels: Number of channels
        num_rates: Number of rate points
    """

    def __init__(self, channels: int, num_rates: int):
        super().__init__()
        self.channels = channels
        self.num_rates = num_rates

        # Learnable gains for each rate
        # Initialize with increasing values for increasing rates
        gains = torch.linspace(0.5, 2.0, num_rates).unsqueeze(1)
        gains = gains.expand(-1, channels)
        self.gains = nn.Parameter(gains)

    def forward(
        self, x: torch.Tensor, rate_idx: int, inverse: bool = False
    ) -> torch.Tensor:
        """Apply gain.

        Args:
            x: Input tensor [B, C, H, W]
            rate_idx: Index of rate point
            inverse: If True, apply inverse gain

        Returns:
            Scaled tensor
        """
        gain = self.gains[rate_idx].view(1, -1, 1, 1).abs() + 1e-6

        if inverse:
            return x / gain
        else:
            return x * gain


class MultiRateCodec(nn.Module):
    """Multi-rate compression codec.

    Supports multiple rate points with a single model using learnable
    gain units for rate control.

    Args:
        num_rates: Number of rate points
        latent_channels: Number of latent channels
        hyper_channels: Number of hyper-latent channels
        num_filters: Number of intermediate filters
    """

    def __init__(
        self,
        num_rates: int = 6,
        latent_channels: int = 192,
        hyper_channels: int = 128,
        num_filters: int = 128,
    ):
        super().__init__()
        self.num_rates = num_rates
        self.latent_channels = latent_channels

        # Shared encoder/decoder
        self.encoder = AnalysisTransform(3, latent_channels, num_filters)
        self.decoder = SynthesisTransform(3, latent_channels, num_filters)

        # Rate-specific gain units
        self.gain = GainUnit(latent_channels, num_rates)

        # Shared entropy model
        self.entropy_model = EntropyModel(latent_channels, hyper_channels)

    def forward(
        self, x: torch.Tensor, rate_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for training.

        Args:
            x: Input image [B, 3, H, W]
            rate_idx: Index of rate point (0 = lowest rate)

        Returns:
            x_hat: Reconstructed image
            losses: Dictionary of loss components
        """
        # Encode
        y = self.encoder(x)

        # Apply rate-specific gain
        y_scaled = self.gain(y, rate_idx, inverse=False)

        # Entropy modeling and quantization
        y_hat, z_hat, likelihoods = self.entropy_model(y_scaled)

        # Inverse gain
        y_hat_unscaled = self.gain(y_hat, rate_idx, inverse=True)

        # Decode
        x_hat = self.decoder(y_hat_unscaled)

        # Compute losses
        losses = self._compute_losses(x, x_hat, likelihoods)

        return x_hat, losses

    def forward_rate(
        self, x: torch.Tensor, rate_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward with specific rate point (alias for forward)."""
        return self.forward(x, rate_idx)

    def _compute_losses(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses."""
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        y_bits = -torch.log2(likelihoods["y"].clamp(min=1e-9)).sum() / num_pixels
        z_bits = -torch.log2(likelihoods["z"].clamp(min=1e-9)).sum() / num_pixels
        bpp = y_bits + z_bits

        return {
            "mse": mse,
            "psnr": psnr,
            "bpp": bpp,
            "y_bpp": y_bits,
            "z_bpp": z_bits,
            "y": likelihoods["y"],
            "z": likelihoods["z"],
        }

    def compress(
        self, x: torch.Tensor, rate_idx: int = 0
    ) -> Tuple[bytes, Tuple[int, ...]]:
        """Compress image at specified rate.

        Args:
            x: Input image [1, 3, H, W]
            rate_idx: Rate point index

        Returns:
            Tuple of (bitstream, original_shape)
        """
        # This would use the same compression as NeuralCompressionCodec
        # but with rate-specific gain applied
        raise NotImplementedError("Use NeuralCompressionCodec.compress for now")

    def decompress(
        self, bitstream: bytes, shape: Tuple[int, ...], rate_idx: int = 0
    ) -> torch.Tensor:
        """Decompress bitstream at specified rate.

        Args:
            bitstream: Compressed data
            shape: Original shape
            rate_idx: Rate point index

        Returns:
            Reconstructed image
        """
        raise NotImplementedError("Use NeuralCompressionCodec.decompress for now")

    def get_rate_distortion_points(
        self, x: torch.Tensor
    ) -> List[Tuple[float, float, float]]:
        """Get R-D points for all rate levels.

        Args:
            x: Test image [1, 3, H, W]

        Returns:
            List of (rate_idx, bpp, psnr) tuples
        """
        self.eval()
        points = []

        with torch.no_grad():
            for rate_idx in range(self.num_rates):
                x_hat, losses = self.forward(x, rate_idx)
                points.append(
                    (rate_idx, losses["bpp"].item(), losses["psnr"].item())
                )

        return points


class ScaleHyperpriorCodec(nn.Module):
    """Scale Hyperprior codec (Ballé et al. 2018).

    Uses hyperprior to predict only scale (not mean),
    simpler than mean-scale hyperprior.

    Args:
        latent_channels: Number of latent channels
        hyper_channels: Number of hyper-latent channels
    """

    def __init__(
        self,
        latent_channels: int = 192,
        hyper_channels: int = 128,
        num_filters: int = 128,
    ):
        super().__init__()

        self.encoder = AnalysisTransform(3, latent_channels, num_filters)
        self.decoder = SynthesisTransform(3, latent_channels, num_filters)

        # Hyper encoder
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )

        # Hyper decoder (only predicts scale)
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, latent_channels, 3, stride=1, padding=1),
        )

        # Factorized prior for z
        self.z_prior_log_scale = nn.Parameter(torch.zeros(1, hyper_channels, 1, 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass."""
        # Encode
        y = self.encoder(x)

        # Hyper-encode
        z = self.hyper_encoder(torch.abs(y))

        # Quantize z
        if self.training:
            z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_hat = torch.round(z)

        # Hyper-decode to get scale
        log_scale = self.hyper_decoder(z_hat)
        scale = torch.exp(log_scale).clamp(min=0.11)

        # Quantize y (zero mean assumed)
        if self.training:
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)

        # Compute likelihoods
        # y likelihood: Gaussian with mean=0, scale from hyperprior
        y_likelihood = self._gaussian_likelihood(y_hat, scale)

        # z likelihood: factorized prior
        z_scale = torch.exp(self.z_prior_log_scale) + 1e-6
        z_likelihood = self._logistic_likelihood(z_hat, z_scale)

        # Decode
        x_hat = self.decoder(y_hat)

        # Compute losses
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        y_bpp = -torch.log2(y_likelihood.clamp(min=1e-9)).sum() / num_pixels
        z_bpp = -torch.log2(z_likelihood.clamp(min=1e-9)).sum() / num_pixels

        return x_hat, {
            "mse": mse,
            "psnr": psnr,
            "bpp": y_bpp + z_bpp,
            "y_bpp": y_bpp,
            "z_bpp": z_bpp,
            "y": y_likelihood,
            "z": z_likelihood,
        }

    def _gaussian_likelihood(
        self, y: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian likelihood with zero mean."""
        import math

        upper = (y + 0.5) / scale
        lower = (y - 0.5) / scale
        upper_cdf = 0.5 * (1 + torch.erf(upper / math.sqrt(2)))
        lower_cdf = 0.5 * (1 + torch.erf(lower / math.sqrt(2)))
        return (upper_cdf - lower_cdf).clamp(min=1e-9)

    def _logistic_likelihood(
        self, z: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Logistic likelihood."""
        upper = torch.sigmoid((z + 0.5) / scale)
        lower = torch.sigmoid((z - 0.5) / scale)
        return (upper - lower).clamp(min=1e-9)
