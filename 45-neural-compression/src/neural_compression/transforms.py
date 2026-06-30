"""Analysis and Synthesis Transforms for Neural Compression.

This module implements:
- GDN: Generalized Divisive Normalization
- AnalysisTransform: Encoder (image -> latent)
- SynthesisTransform: Decoder (latent -> image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GDN(nn.Module):
    """Generalized Divisive Normalization for compression.

    GDN is a key component in neural compression that normalizes
    activations to have more Gaussian-like distributions, improving
    entropy coding efficiency.

    Args:
        num_channels: Number of input channels
        inverse: If True, apply inverse GDN (for decoder)
        beta_min: Minimum value for beta to ensure numerical stability
        gamma_init: Initial value for gamma diagonal
    """

    def __init__(
        self,
        num_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.inverse = inverse
        self.num_channels = num_channels
        self.beta_min = beta_min

        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(num_channels))
        self.gamma = nn.Parameter(torch.eye(num_channels) * gamma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GDN or inverse GDN.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Normalized tensor of same shape
        """
        # Ensure positive parameters
        gamma = self.gamma.abs() + self.beta_min
        beta = self.beta.abs() + self.beta_min

        # Compute normalization factor
        # norm[i] = sqrt(beta[i] + sum_j(gamma[i,j] * x[j]^2))
        x_sq = x ** 2

        # Apply convolution to compute weighted sum of squared activations
        # Reshape gamma for 1x1 convolution: [C_out, C_in, 1, 1]
        gamma_kernel = gamma.unsqueeze(-1).unsqueeze(-1)
        norm = beta.view(1, -1, 1, 1) + F.conv2d(x_sq, gamma_kernel)
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, inverse={self.inverse}"


class AnalysisTransform(nn.Module):
    """Encoder network: Image -> Latent representation.

    Applies strided convolutions with GDN to transform input images
    into a compact latent representation suitable for compression.

    Args:
        in_channels: Number of input channels (3 for RGB)
        latent_channels: Number of latent channels
        num_filters: Number of intermediate filter channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
        num_filters: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_filters = num_filters

        # Each conv layer downsamples by factor of 2
        # Total downsampling: 16x
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 5, stride=2, padding=2),
            GDN(num_filters),
            nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2),
            GDN(num_filters),
            nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2),
            GDN(num_filters),
            nn.Conv2d(num_filters, latent_channels, 5, stride=2, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Latent representation [B, latent_channels, H/16, W/16]
        """
        return self.layers(x)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Compute output shape given input shape.

        Args:
            input_shape: (H, W) tuple

        Returns:
            (H_out, W_out) tuple
        """
        h, w = input_shape
        # 4 layers with stride 2 each -> downsample by 16
        return (h // 16, w // 16)


class SynthesisTransform(nn.Module):
    """Decoder network: Latent -> Reconstructed image.

    Applies transposed convolutions with inverse GDN to transform
    latent representations back to images.

    Args:
        out_channels: Number of output channels (3 for RGB)
        latent_channels: Number of latent channels
        num_filters: Number of intermediate filter channels
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 192,
        num_filters: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_filters = num_filters

        # Each transposed conv layer upsamples by factor of 2
        # Total upsampling: 16x
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, num_filters, 5, stride=2, padding=2, output_padding=1
            ),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(
                num_filters, num_filters, 5, stride=2, padding=2, output_padding=1
            ),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(
                num_filters, num_filters, 5, stride=2, padding=2, output_padding=1
            ),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(
                num_filters, out_channels, 5, stride=2, padding=2, output_padding=1
            ),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            y: Latent representation [B, latent_channels, H/16, W/16]

        Returns:
            Reconstructed image [B, 3, H, W]
        """
        return self.layers(y)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Compute output shape given input shape.

        Args:
            input_shape: (H, W) tuple of latent spatial dimensions

        Returns:
            (H_out, W_out) tuple
        """
        h, w = input_shape
        # 4 layers with stride 2 each -> upsample by 16
        return (h * 16, w * 16)


class ResidualBlock(nn.Module):
    """Residual block with optional GDN normalization.

    Args:
        channels: Number of input/output channels
        use_gdn: Whether to use GDN (True) or batch norm (False)
    """

    def __init__(self, channels: int, use_gdn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        if use_gdn:
            self.norm1 = GDN(channels)
            self.norm2 = GDN(channels)
        else:
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(self.conv1(x))
        out = self.norm2(self.conv2(out))
        return out + residual


class EnhancedAnalysisTransform(nn.Module):
    """Enhanced encoder with residual blocks.

    Args:
        in_channels: Number of input channels
        latent_channels: Number of latent channels
        num_filters: Number of intermediate filter channels
        num_residual_blocks: Number of residual blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
        num_filters: int = 128,
        num_residual_blocks: int = 3,
    ):
        super().__init__()

        # Initial downsampling
        self.conv1 = nn.Conv2d(in_channels, num_filters, 5, stride=2, padding=2)
        self.gdn1 = GDN(num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = GDN(num_filters)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Final downsampling
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = GDN(num_filters)

        self.conv4 = nn.Conv2d(num_filters, latent_channels, 5, stride=2, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.residual_blocks(x)
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x
