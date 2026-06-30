"""Entropy models for neural compression."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import special

logger = logging.getLogger(__name__)


class EntropyModel:
    """Base class for entropy models."""

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """Compute likelihood of values."""
        raise NotImplementedError

    def bits(self, x: np.ndarray) -> float:
        """Compute bits needed to encode."""
        prob = self.likelihood(x)
        return -np.sum(np.log2(prob + 1e-10))

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from the distribution."""
        raise NotImplementedError


class GaussianModel(EntropyModel):
    """
    Gaussian entropy model.

    Models latents as independent Gaussians.
    """

    def __init__(
        self,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9
    ):
        self.scale_bound = scale_bound
        self.tail_mass = tail_mass

    def likelihood(
        self,
        x: np.ndarray,
        scales: np.ndarray,
        means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Gaussian likelihood.

        Uses CDF to compute probability of quantization bins.
        """
        scales = np.maximum(scales, self.scale_bound)
        if means is None:
            means = np.zeros_like(x)

        # Quantization bin edges
        upper = x + 0.5
        lower = x - 0.5

        # CDF values
        upper_cdf = self._standardized_cumulative((upper - means) / scales)
        lower_cdf = self._standardized_cumulative((lower - means) / scales)

        # Likelihood is CDF difference
        likelihood = upper_cdf - lower_cdf

        return np.maximum(likelihood, self.tail_mass)

    def _standardized_cumulative(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + special.erf(x / np.sqrt(2)))

    def sample(
        self,
        shape: Tuple[int, ...],
        scales: np.ndarray,
        means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sample from Gaussian and quantize."""
        if means is None:
            means = np.zeros(shape)

        samples = np.random.normal(means, scales)
        return np.round(samples)


class GMMModel(EntropyModel):
    """
    Gaussian Mixture Model for entropy estimation.

    More flexible than single Gaussian.
    """

    def __init__(
        self,
        num_components: int = 3,
        scale_bound: float = 0.11
    ):
        self.num_components = num_components
        self.scale_bound = scale_bound

    def likelihood(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray
    ) -> np.ndarray:
        """
        Compute GMM likelihood.

        Args:
            x: Input values
            weights: Mixture weights (K,)
            means: Component means (K,)
            scales: Component scales (K,)
        """
        scales = np.maximum(scales, self.scale_bound)
        likelihood = np.zeros_like(x)

        for k in range(self.num_components):
            # Component likelihood
            upper_cdf = 0.5 * (1 + special.erf((x + 0.5 - means[k]) / (scales[k] * np.sqrt(2))))
            lower_cdf = 0.5 * (1 + special.erf((x - 0.5 - means[k]) / (scales[k] * np.sqrt(2))))

            likelihood += weights[k] * (upper_cdf - lower_cdf)

        return np.maximum(likelihood, 1e-9)


class LaplacianModel(EntropyModel):
    """
    Laplacian entropy model.

    Better models heavy-tailed distributions.
    """

    def __init__(self, scale_bound: float = 0.01):
        self.scale_bound = scale_bound

    def likelihood(
        self,
        x: np.ndarray,
        scales: np.ndarray,
        means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Laplacian likelihood."""
        scales = np.maximum(scales, self.scale_bound)
        if means is None:
            means = np.zeros_like(x)

        # Laplacian CDF
        def laplace_cdf(t):
            return np.where(
                t <= 0,
                0.5 * np.exp(t),
                1 - 0.5 * np.exp(-t)
            )

        upper = laplace_cdf((x + 0.5 - means) / scales)
        lower = laplace_cdf((x - 0.5 - means) / scales)

        return np.maximum(upper - lower, 1e-9)


class FullyFactorized(EntropyModel):
    """
    Fully factorized entropy model.

    Uses learnable CDFs for each channel.
    """

    def __init__(
        self,
        num_channels: int,
        num_filters: Tuple[int, ...] = (3, 3, 3)
    ):
        self.num_channels = num_channels
        self.num_filters = num_filters

        # CDF parameters (simplified)
        self.matrices = []
        self.biases = []

        prev_dim = 1
        for filt in num_filters:
            self.matrices.append(
                np.random.randn(num_channels, prev_dim, filt) * 0.1
            )
            self.biases.append(
                np.random.randn(num_channels, filt) * 0.1
            )
            prev_dim = filt

        # Final layer
        self.matrices.append(
            np.random.randn(num_channels, prev_dim, 1) * 0.1
        )
        self.biases.append(
            np.random.randn(num_channels, 1) * 0.1
        )

    def _logits_cumulative(self, x: np.ndarray, channel: int) -> np.ndarray:
        """Compute CDF logits for a channel."""
        # x shape: (num_samples,)
        h = x.reshape(-1, 1)

        for mat, bias in zip(self.matrices, self.biases):
            h = h @ mat[channel] + bias[channel]
            h = h + np.tanh(h)  # Non-linearity

        return h.squeeze()

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for all channels.

        Args:
            x: Quantized values (..., channels)

        Returns:
            Likelihoods
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.num_channels)
        likelihood = np.zeros_like(x_flat)

        for c in range(self.num_channels):
            upper = self._logits_cumulative(x_flat[:, c] + 0.5, c)
            lower = self._logits_cumulative(x_flat[:, c] - 0.5, c)

            # Sigmoid to get CDF
            upper_cdf = 1 / (1 + np.exp(-upper))
            lower_cdf = 1 / (1 + np.exp(-lower))

            likelihood[:, c] = upper_cdf - lower_cdf

        return np.maximum(likelihood.reshape(shape), 1e-9)


class ConditionalGaussian(EntropyModel):
    """
    Conditional Gaussian model.

    Mean and scale predicted from side information.
    """

    def __init__(self, latent_channels: int):
        self.latent_channels = latent_channels
        self.gaussian = GaussianModel()

    def likelihood(
        self,
        x: np.ndarray,
        context: np.ndarray
    ) -> np.ndarray:
        """
        Compute conditional likelihood.

        Args:
            x: Latent values
            context: Context from hyperprior

        Returns:
            Likelihoods
        """
        # Context provides scales and means
        # Simplified: split context
        scales = np.abs(context[..., :self.latent_channels]) + 0.01
        means = context[..., self.latent_channels:] if context.shape[-1] > self.latent_channels else None

        return self.gaussian.likelihood(x, scales, means)


class AutoregressiveModel(EntropyModel):
    """
    Autoregressive entropy model.

    Context from previous latents.
    """

    def __init__(self, latent_channels: int, context_size: int = 5):
        self.latent_channels = latent_channels
        self.context_size = context_size

        # Context network (simplified)
        self.context_weight = np.random.randn(
            context_size ** 2, latent_channels * 2
        ) * 0.02

    def likelihood(
        self,
        x: np.ndarray,
        hyper_params: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Compute autoregressive likelihood.

        Combines hyperprior params with spatial context.
        """
        hyper_scales, hyper_means = hyper_params
        batch, height, width, channels = x.shape

        likelihood = np.ones_like(x)
        gaussian = GaussianModel()

        # Process in raster scan order
        for i in range(height):
            for j in range(width):
                # Get context from previous positions
                context = self._get_context(x, i, j)

                # Predict parameters
                context_params = context @ self.context_weight

                # Combine with hyperprior
                scales = hyper_scales[:, i, j, :] + np.abs(context_params[:, :channels])
                means = hyper_means[:, i, j, :] + context_params[:, channels:]

                # Likelihood
                likelihood[:, i, j, :] = gaussian.likelihood(
                    x[:, i, j, :], scales, means
                )

        return likelihood

    def _get_context(
        self,
        x: np.ndarray,
        i: int,
        j: int
    ) -> np.ndarray:
        """Get context from previous positions."""
        batch, height, width, channels = x.shape
        context_size = self.context_size
        half = context_size // 2

        context = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                if di < 0 or (di == 0 and dj < 0):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        context.append(x[:, ni, nj, 0])  # Simplified
                    else:
                        context.append(np.zeros(batch))

        return np.stack(context, axis=-1)


def estimate_entropy(
    x: np.ndarray,
    model: EntropyModel,
    **kwargs
) -> float:
    """
    Estimate entropy of data under model.

    Args:
        x: Data
        model: Entropy model
        **kwargs: Model-specific parameters

    Returns:
        Entropy in bits
    """
    likelihood = model.likelihood(x, **kwargs)
    return -np.sum(np.log2(likelihood + 1e-10))


def kl_divergence(
    p_likelihood: np.ndarray,
    q_likelihood: np.ndarray
) -> float:
    """
    Compute KL divergence between distributions.

    KL(p || q) = sum(p * log(p/q))
    """
    p = p_likelihood / p_likelihood.sum()
    q = q_likelihood / q_likelihood.sum()

    return float(np.sum(p * np.log(p / (q + 1e-10) + 1e-10)))
