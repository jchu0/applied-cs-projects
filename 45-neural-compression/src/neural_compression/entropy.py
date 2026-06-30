"""Entropy Models for Neural Compression.

This module implements:
- FactorizedPrior: Learned prior for hyper-latents
- HyperAnalysis/HyperSynthesis: Hyperprior networks
- EntropyModel: Full entropy model with hyperprior
- GaussianEntropyModel: Gaussian conditional entropy model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EntropyOutput:
    """Output from entropy model forward pass."""

    y_hat: torch.Tensor  # Quantized main latent
    z_hat: torch.Tensor  # Quantized hyper-latent
    y_likelihood: torch.Tensor  # Likelihood of y
    z_likelihood: torch.Tensor  # Likelihood of z
    mean: torch.Tensor  # Predicted mean for y
    scale: torch.Tensor  # Predicted scale for y


class FactorizedPrior(nn.Module):
    """Factorized prior for hyper-latents.

    Uses a learned non-parametric distribution for each channel.
    The CDF is modeled using a series of monotonic transforms.

    Args:
        channels: Number of channels
        init_scale: Initial scale for CDF parameters
        num_filters: Number of filters in the transform layers
    """

    def __init__(
        self,
        channels: int,
        init_scale: float = 10.0,
        num_filters: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale
        self.num_filters = num_filters

        # Learnable CDF parameters
        # Using a simplified parametric approach with logistic distributions
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # For more complex learned CDFs (optional advanced version)
        self.matrices = nn.ParameterList(
            [
                nn.Parameter(torch.eye(channels).unsqueeze(0) * init_scale)
                for _ in range(num_filters)
            ]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, channels, 1)) for _ in range(num_filters)]
        )

    def forward(
        self, z: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and compute likelihood.

        Args:
            z: Input tensor [B, C, H, W]
            training: Whether in training mode (use noise) or eval (use rounding)

        Returns:
            z_hat: Quantized values
            likelihood: Probability of each symbol
        """
        if training:
            # Add uniform noise for differentiable approximation
            z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_hat = torch.round(z)

        # Compute likelihood using logistic distribution
        likelihood = self._compute_likelihood(z_hat)

        return z_hat, likelihood

    def _compute_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """Compute likelihood using learned logistic distribution.

        Args:
            z: Quantized tensor [B, C, H, W]

        Returns:
            Likelihood tensor of same shape
        """
        # Use logistic distribution
        scale = torch.exp(self.log_scale) + 1e-6
        loc = self.loc

        # Integrate logistic PDF over quantization bin [-0.5, 0.5]
        centered = z - loc
        upper = torch.sigmoid((centered + 0.5) / scale)
        lower = torch.sigmoid((centered - 0.5) / scale)

        likelihood = upper - lower
        return likelihood.clamp(min=1e-9)

    def get_cdf(self, z: torch.Tensor, num_symbols: int = 256) -> torch.Tensor:
        """Get CDF for arithmetic coding.

        Args:
            z: Tensor to get CDFs for
            num_symbols: Number of quantization levels

        Returns:
            CDFs tensor [batch, channels, height, width, num_symbols+1]
        """
        scale = torch.exp(self.log_scale) + 1e-6

        # Create symbol range
        symbols = torch.arange(num_symbols + 1, device=z.device, dtype=z.dtype)
        symbols = symbols - num_symbols // 2

        # Compute CDF at each symbol boundary
        # Shape: [1, C, 1, 1, num_symbols+1]
        symbols = symbols.view(1, 1, 1, 1, -1)
        cdf = torch.sigmoid((symbols - 0.5) / scale.unsqueeze(-1))

        return cdf


class HyperAnalysis(nn.Module):
    """Hyper-encoder for entropy parameters.

    Maps latent representation to hyper-latents that capture
    spatial dependencies in the entropy model.

    Args:
        latent_channels: Number of main latent channels
        hyper_channels: Number of hyper-latent channels
    """

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels

        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Encode latents to hyper-latents.

        Args:
            y: Latent representation [B, C, H, W]

        Returns:
            Hyper-latent [B, hyper_channels, H/4, W/4]
        """
        return self.layers(torch.abs(y))


class HyperSynthesis(nn.Module):
    """Hyper-decoder for entropy parameters.

    Reconstructs entropy parameters (mean, scale) from hyper-latents.

    Args:
        latent_channels: Number of main latent channels
        hyper_channels: Number of hyper-latent channels
    """

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels

        self.deconv1 = nn.ConvTranspose2d(
            hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(hyper_channels, latent_channels * 2, 3, stride=1, padding=1)

    def forward(
        self, z: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hyper-latents to entropy parameters.

        Args:
            z: Hyper-latent [B, hyper_channels, H/4, W/4]
            target_size: Optional (H, W) target spatial size for the output.
                         If provided, output will be resized to match.

        Returns:
            mean: Predicted mean [B, latent_channels, H, W]
            scale: Predicted scale [B, latent_channels, H, W]
        """
        out = self.relu1(self.deconv1(z))
        out = self.relu2(self.deconv2(out))
        out = self.conv_out(out)

        # Resize to target size if needed (handles non-square images)
        if target_size is not None:
            h, w = target_size
            if out.shape[2] != h or out.shape[3] != w:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        mean, log_scale = out.chunk(2, dim=1)
        # Ensure positive scale with minimum value
        scale = torch.exp(log_scale).clamp(min=0.11)
        return mean, scale


class GaussianEntropyModel(nn.Module):
    """Gaussian conditional entropy model.

    Computes likelihoods under a Gaussian distribution given mean and scale.
    Uses integration over quantization bins for accurate likelihood computation.
    """

    def __init__(self, tail_mass: float = 1e-9):
        super().__init__()
        self.tail_mass = tail_mass

    def forward(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and compute likelihood.

        Args:
            y: Input latent [B, C, H, W]
            mean: Predicted mean [B, C, H, W]
            scale: Predicted scale [B, C, H, W]
            training: Whether in training mode

        Returns:
            y_hat: Quantized latent
            likelihood: Probability of each symbol
        """
        if training:
            # Add uniform noise for differentiable approximation
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)

        likelihood = self._gaussian_likelihood(y_hat, mean, scale)
        return y_hat, likelihood

    def _gaussian_likelihood(
        self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute likelihood under Gaussian distribution.

        Integrates the Gaussian PDF over quantization bin [y-0.5, y+0.5].

        Args:
            y: Quantized values
            mean: Distribution mean
            scale: Distribution scale (std dev)

        Returns:
            Likelihood tensor
        """
        half = 0.5
        # Standardize
        upper = (y + half - mean) / scale
        lower = (y - half - mean) / scale

        # Standard normal CDF
        upper_cdf = self._standardized_cumulative(upper)
        lower_cdf = self._standardized_cumulative(lower)

        likelihood = upper_cdf - lower_cdf
        return likelihood.clamp(min=self.tail_mass)

    def _standardized_cumulative(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal CDF using error function.

        Args:
            x: Input tensor

        Returns:
            CDF values
        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def get_cdf(
        self, mean: torch.Tensor, scale: torch.Tensor, num_symbols: int = 256
    ) -> torch.Tensor:
        """Get CDFs for arithmetic coding.

        Args:
            mean: Distribution means [B, C, H, W]
            scale: Distribution scales [B, C, H, W]
            num_symbols: Number of quantization levels

        Returns:
            CDFs tensor [B, C, H, W, num_symbols+1]
        """
        # Create symbol range centered at 0
        symbols = torch.arange(num_symbols + 1, device=mean.device, dtype=mean.dtype)
        symbols = symbols - num_symbols // 2

        # Expand for broadcasting
        symbols = symbols.view(1, 1, 1, 1, -1)
        mean_exp = mean.unsqueeze(-1)
        scale_exp = scale.unsqueeze(-1)

        # Compute CDF at each symbol boundary (symbol - 0.5)
        x = (symbols - 0.5 - mean_exp) / scale_exp
        cdf = self._standardized_cumulative(x)

        return cdf


class EntropyModel(nn.Module):
    """Entropy model with hyperprior.

    Combines hyperprior analysis/synthesis with Gaussian entropy model
    to estimate probability distributions for arithmetic coding.

    Args:
        latent_channels: Number of main latent channels
        hyper_channels: Number of hyper-latent channels
    """

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels

        self.hyper_analysis = HyperAnalysis(latent_channels, hyper_channels)
        self.hyper_synthesis = HyperSynthesis(latent_channels, hyper_channels)

        # For quantized hyper-latents
        self.hyper_entropy = FactorizedPrior(hyper_channels)

        # For main latents given hyper-latent derived parameters
        self.gaussian_entropy = GaussianEntropyModel()

    def forward(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute entropy parameters and estimate bits.

        Args:
            y: Latent representation [B, C, H, W]

        Returns:
            y_hat: Quantized latent
            z_hat: Quantized hyper-latent
            likelihoods: Dictionary of likelihoods for rate computation
        """
        # Hyper-encoder
        z = self.hyper_analysis(y)

        # Quantize hyper-latent
        z_hat, z_likelihood = self.hyper_entropy(z, training=self.training)

        # Hyper-decoder - pass target size to handle non-square images
        target_size = (y.shape[2], y.shape[3])
        mean, scale = self.hyper_synthesis(z_hat, target_size=target_size)

        # Quantize main latent
        y_hat, y_likelihood = self.gaussian_entropy(
            y, mean, scale, training=self.training
        )

        return y_hat, z_hat, {"y": y_likelihood, "z": z_likelihood}

    def get_entropy_parameters(
        self, z_hat: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get entropy parameters from decoded hyper-latent.

        Args:
            z_hat: Quantized hyper-latent
            target_size: Optional (H, W) target spatial size

        Returns:
            mean: Predicted mean
            scale: Predicted scale
        """
        return self.hyper_synthesis(z_hat, target_size=target_size)

    def compress(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare tensors for compression.

        Args:
            y: Latent to compress

        Returns:
            y_hat: Quantized y
            z_hat: Quantized z
            mean: Entropy mean parameter
            scale: Entropy scale parameter
        """
        with torch.no_grad():
            z = self.hyper_analysis(y)
            z_hat = torch.round(z)
            mean, scale = self.hyper_synthesis(z_hat)
            y_hat = torch.round(y)
        return y_hat, z_hat, mean, scale

    def compute_bits(self, likelihoods: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute bits from likelihoods.

        Args:
            likelihoods: Dictionary with 'y' and 'z' likelihoods

        Returns:
            Dictionary with bit counts
        """
        y_bits = -torch.log2(likelihoods["y"]).sum().item()
        z_bits = -torch.log2(likelihoods["z"]).sum().item()

        return {"y_bits": y_bits, "z_bits": z_bits, "total_bits": y_bits + z_bits}


class MeanScaleHyperprior(nn.Module):
    """Mean-Scale Hyperprior model (Minnen et al. 2018).

    A more powerful hyperprior that predicts both mean and scale,
    improving compression efficiency.

    Args:
        latent_channels: Number of main latent channels
        hyper_channels: Number of hyper-latent channels
    """

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 192):
        super().__init__()
        self.latent_channels = latent_channels

        # Hyper encoder
        self.ha = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )

        # Hyper decoder
        self.hs_deconv1 = nn.ConvTranspose2d(
            hyper_channels, latent_channels, 5, stride=2, padding=2, output_padding=1
        )
        self.hs_relu1 = nn.LeakyReLU(inplace=True)
        self.hs_deconv2 = nn.ConvTranspose2d(
            latent_channels,
            latent_channels * 3 // 2,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.hs_relu2 = nn.LeakyReLU(inplace=True)
        self.hs_conv_out = nn.Conv2d(latent_channels * 3 // 2, latent_channels * 2, 3, padding=1)

        self.factorized = FactorizedPrior(hyper_channels)
        self.gaussian = GaussianEntropyModel()

    def _hyper_decode(
        self, z_hat: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Decode hyper-latent to parameters with optional resizing."""
        out = self.hs_relu1(self.hs_deconv1(z_hat))
        out = self.hs_relu2(self.hs_deconv2(out))
        out = self.hs_conv_out(out)

        # Resize to target size if needed (handles non-square images)
        if target_size is not None:
            h, w = target_size
            if out.shape[2] != h or out.shape[3] != w:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out

    def forward(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with entropy estimation.

        Args:
            y: Main latent [B, C, H, W]

        Returns:
            y_hat: Quantized main latent
            z_hat: Quantized hyper-latent
            likelihoods: Dictionary with y and z likelihoods
        """
        z = self.ha(y)
        z_hat, z_likelihood = self.factorized(z, self.training)

        # Decode with target size to handle non-square images
        target_size = (y.shape[2], y.shape[3])
        params = self._hyper_decode(z_hat, target_size=target_size)
        mean, log_scale = params.chunk(2, dim=1)
        scale = torch.exp(log_scale).clamp(min=0.11)

        y_hat, y_likelihood = self.gaussian(y, mean, scale, self.training)

        return y_hat, z_hat, {"y": y_likelihood, "z": z_likelihood}
