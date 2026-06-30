"""Core compression primitives for neural codecs."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class CompressedData:
    """Container for compressed data."""
    latent_shape: Tuple[int, ...]
    strings: List[bytes]        # Entropy coded strings
    side_info: Optional[bytes] = None  # Hyperprior side info

    @property
    def total_bits(self) -> int:
        total = sum(len(s) * 8 for s in self.strings)
        if self.side_info:
            total += len(self.side_info) * 8
        return total


@dataclass
class CompressionMetrics:
    """Metrics for compression quality."""
    bpp: float           # Bits per pixel
    psnr: float          # Peak Signal-to-Noise Ratio
    ms_ssim: float       # Multi-Scale SSIM
    rate: float          # Rate (bits)
    distortion: float    # Distortion (MSE)


def quantize(x: np.ndarray, mode: str = 'round') -> np.ndarray:
    """
    Quantize values.

    Args:
        x: Input values
        mode: 'round', 'noise', or 'ste'

    Returns:
        Quantized values
    """
    if mode == 'round':
        return np.round(x)
    elif mode == 'noise':
        # Add uniform noise for training
        noise = np.random.uniform(-0.5, 0.5, x.shape)
        return x + noise
    elif mode == 'ste':
        # Straight-through estimator
        return np.round(x)  # Forward: round, backward: identity
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def dequantize(x: np.ndarray) -> np.ndarray:
    """Dequantize (identity for neural codecs)."""
    return x.astype(np.float32)


def calculate_rate(prob: np.ndarray) -> float:
    """
    Calculate rate from probability.

    Rate = -log2(prob)
    """
    prob = np.clip(prob, 1e-10, 1.0)
    return -np.sum(np.log2(prob))


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        original: Original image
        reconstructed: Reconstructed image

    Returns:
        PSNR in dB
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')

    max_val = 255.0 if original.max() > 1 else 1.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ms_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_val: float = 1.0
) -> float:
    """
    Calculate Multi-Scale Structural Similarity.

    Simplified implementation.
    """
    def ssim(x, y, K1=0.01, K2=0.03):
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2

        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.std(x)
        sigma_y = np.std(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))

        l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
        c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
        s = (sigma_xy + C2 / 2) / (sigma_x * sigma_y + C2 / 2)

        return l * c * s

    # Multi-scale
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    levels = len(weights)

    ms_ssim_val = 1.0
    for i in range(levels):
        ss = ssim(original, reconstructed)
        if i < levels - 1:
            ms_ssim_val *= ss ** weights[i]
            # Downsample
            original = original[::2, ::2]
            reconstructed = reconstructed[::2, ::2]
        else:
            ms_ssim_val *= ss ** weights[i]

    return ms_ssim_val


def rate_distortion_loss(
    rate: float,
    distortion: float,
    lmbda: float
) -> float:
    """
    Rate-distortion loss.

    L = R + lambda * D
    """
    return rate + lmbda * distortion


class GDN:
    """
    Generalized Divisive Normalization.

    Used in neural image compression for local gain control.
    """

    def __init__(self, num_channels: int, inverse: bool = False):
        self.num_channels = num_channels
        self.inverse = inverse

        # Parameters
        self.beta = np.ones(num_channels) * 1e-6
        self.gamma = np.eye(num_channels) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply GDN.

        Args:
            x: Input (batch, height, width, channels)

        Returns:
            Normalized output
        """
        # Compute normalization factor
        x_sq = x ** 2
        norm = np.sqrt(self.beta + np.tensordot(x_sq, self.gamma, axes=[[3], [0]]))

        if self.inverse:
            return x * norm
        else:
            return x / norm


class Analysis:
    """Analysis transform (encoder)."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
        num_filters: int = 192
    ):
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_filters = num_filters

        # Convolutional layers (simplified)
        self.conv1 = np.random.randn(5, 5, in_channels, num_filters) * 0.02
        self.conv2 = np.random.randn(5, 5, num_filters, num_filters) * 0.02
        self.conv3 = np.random.randn(5, 5, num_filters, num_filters) * 0.02
        self.conv4 = np.random.randn(5, 5, num_filters, latent_channels) * 0.02

        self.gdn1 = GDN(num_filters)
        self.gdn2 = GDN(num_filters)
        self.gdn3 = GDN(num_filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Encode image to latent representation.

        Args:
            x: Input image (batch, height, width, channels)

        Returns:
            Latent representation
        """
        # Simplified - actual would use proper convolution
        h = x
        for i in range(4):
            # Placeholder for conv + downsample
            h = h[:, ::2, ::2, :]
            if i < 3:
                # Ensure channel dimension matches
                if h.shape[-1] != self.num_filters:
                    h = np.concatenate([h] * (self.num_filters // h.shape[-1] + 1), axis=-1)
                    h = h[..., :self.num_filters]

        # Final channel adjustment
        if h.shape[-1] != self.latent_channels:
            h = np.concatenate([h] * (self.latent_channels // h.shape[-1] + 1), axis=-1)
            h = h[..., :self.latent_channels]

        return h


class Synthesis:
    """Synthesis transform (decoder)."""

    def __init__(
        self,
        latent_channels: int = 192,
        out_channels: int = 3,
        num_filters: int = 192
    ):
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        # Transposed convolutions (simplified)
        self.conv1 = np.random.randn(5, 5, latent_channels, num_filters) * 0.02
        self.conv2 = np.random.randn(5, 5, num_filters, num_filters) * 0.02
        self.conv3 = np.random.randn(5, 5, num_filters, num_filters) * 0.02
        self.conv4 = np.random.randn(5, 5, num_filters, out_channels) * 0.02

        self.igdn1 = GDN(num_filters, inverse=True)
        self.igdn2 = GDN(num_filters, inverse=True)
        self.igdn3 = GDN(num_filters, inverse=True)

    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Decode latent to image.

        Args:
            y: Latent representation

        Returns:
            Reconstructed image
        """
        h = y
        for i in range(4):
            # Upsample
            h = np.repeat(np.repeat(h, 2, axis=1), 2, axis=2)
            if i < 3:
                if h.shape[-1] != self.num_filters:
                    h = np.concatenate([h] * (self.num_filters // h.shape[-1] + 1), axis=-1)
                    h = h[..., :self.num_filters]

        # Final channel adjustment
        if h.shape[-1] != self.out_channels:
            h = h[..., :self.out_channels]

        return h


class HyperAnalysis:
    """Hyper-analysis for side information."""

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 192):
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Encode latent to hyper-latent."""
        # Simplified
        y_abs = np.abs(y)
        z = y_abs[:, ::2, ::2, :]
        z = z[:, ::2, ::2, :]
        return z


class HyperSynthesis:
    """Hyper-synthesis to decode side information."""

    def __init__(self, hyper_channels: int = 192, latent_channels: int = 192):
        self.hyper_channels = hyper_channels
        self.latent_channels = latent_channels

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode hyper-latent to entropy parameters.

        Returns:
            Tuple of (scales, means)
        """
        # Upsample
        h = np.repeat(np.repeat(z, 2, axis=1), 2, axis=2)
        h = np.repeat(np.repeat(h, 2, axis=1), 2, axis=2)

        # Split into scales and means
        scales = np.abs(h) + 0.01
        means = np.zeros_like(h)

        return scales, means
