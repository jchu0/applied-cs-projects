"""Perceptual and distortion losses for neural compression.

This module implements:
- MSELoss: Standard mean squared error
- MS_SSIM: Multi-scale structural similarity
- LPIPSLoss: A VGG-feature perceptual approximation (NOT calibrated LPIPS; see class)
- RateDistortionLoss: Combined R-D loss
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation
        device: Device for the tensor

    Returns:
        2D Gaussian kernel [1, 1, size, size]
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.

    Computes 1 - SSIM, so minimizing this maximizes SSIM.

    Args:
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window
        data_range: Range of input data (e.g., 1.0 for [0,1] images)
        channel: Number of channels
        size_average: If True, return mean loss
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channel: int = 3,
        size_average: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        self.size_average = size_average

        self.register_buffer("window", None)

        # Constants for stability
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def _get_window(self, device: torch.device) -> torch.Tensor:
        """Get or create the Gaussian window."""
        if self.window is None or self.window.device != device:
            window = gaussian_kernel(self.window_size, self.sigma, device)
            self.window = window.expand(self.channel, 1, -1, -1)
        return self.window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            x: Predicted image [B, C, H, W]
            y: Target image [B, C, H, W]

        Returns:
            1 - SSIM (scalar or per-sample)
        """
        window = self._get_window(x.device)

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_xy
        )

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=[1, 2, 3])


class MS_SSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity loss.

    Computes MS-SSIM across multiple scales for better perceptual quality.

    Args:
        window_size: Size of Gaussian window
        data_range: Range of input data
        weights: Weights for each scale (default from original paper)
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

        if weights is None:
            # Default weights from MS-SSIM paper
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        else:
            self.weights = weights

        self.ssim = SSIMLoss(window_size, data_range=data_range)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM loss.

        Args:
            x: Predicted image [B, C, H, W]
            y: Target image [B, C, H, W]

        Returns:
            1 - MS_SSIM
        """
        weights = torch.tensor(self.weights, device=x.device, dtype=x.dtype)

        ms_ssim = 1.0
        for i, weight in enumerate(weights):
            if i > 0:
                # Downsample by 2
                x = F.avg_pool2d(x, 2)
                y = F.avg_pool2d(y, 2)

            ssim_val = 1 - self.ssim(x, y)  # Get SSIM (not 1-SSIM)
            ms_ssim = ms_ssim * (ssim_val ** weight)

        return 1 - ms_ssim


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Uses pre-trained VGG features to compute perceptual similarity.

    Args:
        layers: Which VGG layers to use
        weights: Weights for each layer
        normalize_input: Whether to normalize input to ImageNet stats
    """

    def __init__(
        self,
        layers: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]

        self.layers = layers
        self.weights = weights

        # VGG feature extractor (lazy initialization)
        self.vgg = None
        self.layer_indices = None

    def _init_vgg(self, device: torch.device):
        """Initialize VGG model."""
        try:
            from torchvision.models import vgg19, VGG19_Weights

            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import vgg19

            vgg = vgg19(pretrained=True).features

        vgg = vgg.to(device).eval()

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Map layer names to indices
        layer_name_to_idx = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }

        self.layer_indices = [layer_name_to_idx[l] for l in self.layers]
        self.vgg = vgg

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            x: Predicted image [B, C, H, W]
            y: Target image [B, C, H, W]

        Returns:
            Perceptual loss scalar
        """
        if self.vgg is None:
            self._init_vgg(x.device)

        if self.normalize_input:
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            y = (y - mean) / std

        # Extract features
        loss = 0.0
        x_feat = x
        y_feat = y

        for i, layer in enumerate(self.vgg):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)

            if i in self.layer_indices:
                idx = self.layer_indices.index(i)
                loss = loss + self.weights[idx] * F.mse_loss(x_feat, y_feat)

        return loss


class LPIPSLoss(nn.Module):
    """VGG-feature perceptual loss (an approximation, NOT calibrated LPIPS).

    IMPORTANT: This is *not* the true LPIPS metric of Zhang et al. (2018). Real
    LPIPS uses linear layers calibrated on a large human-perceptual-judgment
    dataset (BAPPS) on top of channelwise-normalized deep features. This class
    is a lightweight stand-in that simply delegates to :class:`VGGPerceptualLoss`
    (feature-space MSE over selected VGG-19 layers) with uniform, *non-learned*
    layer weights. It is a reasonable perceptual training signal but its values
    are not comparable to published LPIPS scores and must not be reported as
    LPIPS. For calibrated LPIPS, install and use the ``lpips`` package.

    Args:
        net: Kept for API familiarity with LPIPS; only ``'vgg'`` is supported
            (the underlying feature extractor is always VGG-19).
        normalize: Whether to apply ImageNet input normalization before features.
    """

    def __init__(self, net: str = "vgg", normalize: bool = True):
        super().__init__()
        if net != "vgg":
            raise ValueError(
                f"Unsupported net {net!r}: this VGG-feature approximation only "
                "supports net='vgg'."
            )
        self.net = net
        self.normalize = normalize

        # Use VGG features
        self.vgg_loss = VGGPerceptualLoss(normalize_input=normalize)

        # Uniform per-layer weights. NOTE: real LPIPS learns these on human
        # perceptual judgments; here they are fixed to 1.0 (i.e. not learned).
        self.register_buffer(
            "scales", torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the VGG-feature perceptual distance.

        Args:
            x: Predicted image [B, C, H, W]
            y: Target image [B, C, H, W]

        Returns:
            Scalar perceptual distance (VGG-feature MSE; not calibrated LPIPS).
        """
        return self.vgg_loss(x, y)


class RateDistortionLoss(nn.Module):
    """Combined rate-distortion loss for neural compression.

    Combines distortion loss (MSE, SSIM, or perceptual) with rate loss (bpp).

    Args:
        lambda_rd: Rate-distortion trade-off parameter
        distortion_type: Type of distortion ('mse', 'ms_ssim', 'lpips', 'mixed')
        lpips_weight: Weight for LPIPS loss when using 'mixed'
    """

    def __init__(
        self,
        lambda_rd: float = 0.01,
        distortion_type: str = "mse",
        lpips_weight: float = 0.1,
    ):
        super().__init__()
        self.lambda_rd = lambda_rd
        self.distortion_type = distortion_type
        self.lpips_weight = lpips_weight

        # Initialize distortion losses
        if distortion_type in ("ms_ssim", "mixed"):
            self.ms_ssim = MS_SSIMLoss()

        if distortion_type in ("lpips", "mixed"):
            self.lpips = LPIPSLoss()

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute rate-distortion loss.

        Args:
            x: Original image [B, C, H, W]
            x_hat: Reconstructed image [B, C, H, W]
            likelihoods: Dictionary with 'y' and 'z' likelihoods

        Returns:
            loss: Total loss
            metrics: Dictionary with loss components
        """
        # Rate
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        y_bpp = -torch.log2(likelihoods["y"].clamp(min=1e-9)).sum() / num_pixels
        z_bpp = -torch.log2(likelihoods["z"].clamp(min=1e-9)).sum() / num_pixels
        bpp = y_bpp + z_bpp

        # Distortion
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        if self.distortion_type == "mse":
            distortion = mse
        elif self.distortion_type == "ms_ssim":
            distortion = self.ms_ssim(x_hat, x)
        elif self.distortion_type == "lpips":
            distortion = self.lpips(x_hat, x)
        elif self.distortion_type == "mixed":
            distortion = mse + self.lpips_weight * self.lpips(x_hat, x)
        else:
            raise ValueError(f"Unknown distortion type: {self.distortion_type}")

        # Total loss
        loss = distortion + self.lambda_rd * bpp

        metrics = {
            "loss": loss,
            "distortion": distortion,
            "mse": mse,
            "psnr": psnr,
            "bpp": bpp,
            "y_bpp": y_bpp,
            "z_bpp": z_bpp,
        }

        return loss, metrics


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1 variant).

    More robust to outliers than MSE.

    Args:
        epsilon: Smoothing parameter
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Loss scalar
        """
        diff = x - y
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))
