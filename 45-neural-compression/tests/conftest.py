"""Pytest fixtures for neural compression tests."""

import pytest
import numpy as np
import importlib.util

# Check if torch is available
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    # Set random seeds for reproducibility
    torch.manual_seed(42)

np.random.seed(42)


def pytest_collection_modifyitems(config, items):
    """Skip tests that require torch if it's not installed."""
    if not TORCH_AVAILABLE:
        skip_torch = pytest.mark.skip(reason="PyTorch not installed")
        for item in items:
            item.add_marker(skip_torch)


@pytest.fixture
def device():
    """Get available device (CPU for tests)."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 2


@pytest.fixture
def image_size():
    """Default image size (must be divisible by 16 for transforms)."""
    return (64, 64)


@pytest.fixture
def latent_channels():
    """Default number of latent channels."""
    return 48  # Smaller for faster tests


@pytest.fixture
def hyper_channels():
    """Default number of hyper-latent channels."""
    return 32  # Smaller for faster tests


@pytest.fixture
def num_filters():
    """Default number of intermediate filters."""
    return 32  # Smaller for faster tests


@pytest.fixture
def random_image(batch_size, image_size):
    """Generate random image tensor [B, 3, H, W] in [0, 1]."""
    h, w = image_size
    return torch.rand(batch_size, 3, h, w)


@pytest.fixture
def single_image(image_size):
    """Generate single random image [1, 3, H, W] in [0, 1]."""
    h, w = image_size
    return torch.rand(1, 3, h, w)


@pytest.fixture
def random_latent(batch_size, latent_channels, image_size):
    """Generate random latent tensor [B, C, H/16, W/16]."""
    h, w = image_size
    return torch.randn(batch_size, latent_channels, h // 16, w // 16)


@pytest.fixture
def random_hyper_latent(batch_size, hyper_channels, image_size):
    """Generate random hyper-latent tensor [B, C, H/64, W/64]."""
    h, w = image_size
    return torch.randn(batch_size, hyper_channels, h // 64, w // 64)


@pytest.fixture
def simple_symbols():
    """Simple symbol sequence for codec testing."""
    return np.array([0, 1, 2, 1, 0, 2, 1, 0], dtype=np.int32)


@pytest.fixture
def simple_cdfs():
    """Simple CDFs for codec testing (uniform distribution)."""
    num_symbols = 8
    vocab_size = 3
    precision = 16
    max_val = 1 << precision

    # Uniform CDF for 3 symbols: [0, 1/3, 2/3, 1] * max_val
    cdfs = np.zeros((num_symbols, vocab_size + 1), dtype=np.int64)
    for i in range(num_symbols):
        cdfs[i] = np.array([0, max_val // 3, 2 * max_val // 3, max_val])

    return cdfs


@pytest.fixture
def gaussian_cdfs():
    """Gaussian-distributed CDFs for more realistic testing."""
    import math

    num_symbols = 16
    vocab_size = 16
    precision = 16
    max_val = 1 << precision

    cdfs = np.zeros((num_symbols, vocab_size + 1), dtype=np.int64)
    for i in range(num_symbols):
        mean = vocab_size // 2
        std = 2.0
        for s in range(vocab_size + 1):
            val = (s - mean - 0.5) / std
            cdf_val = 0.5 * (1 + math.erf(val / math.sqrt(2)))
            cdfs[i, s] = int(cdf_val * max_val)

        # Ensure monotonicity
        for s in range(1, vocab_size + 1):
            if cdfs[i, s] <= cdfs[i, s - 1]:
                cdfs[i, s] = cdfs[i, s - 1] + 1

    return cdfs


@pytest.fixture
def random_gaussian_symbols():
    """Random symbols from Gaussian-like distribution."""
    np.random.seed(42)
    mean = 8
    std = 2
    symbols = np.random.normal(mean, std, 16).astype(np.int32)
    symbols = np.clip(symbols, 0, 15)
    return symbols


# Model fixtures


@pytest.fixture
def gdn_layer(num_filters):
    """GDN layer instance."""
    from src.neural_compression.transforms import GDN

    return GDN(num_filters)


@pytest.fixture
def inverse_gdn_layer(num_filters):
    """Inverse GDN layer instance."""
    from src.neural_compression.transforms import GDN

    return GDN(num_filters, inverse=True)


@pytest.fixture
def analysis_transform(latent_channels, num_filters):
    """Analysis transform (encoder) instance."""
    from src.neural_compression.transforms import AnalysisTransform

    return AnalysisTransform(
        in_channels=3, latent_channels=latent_channels, num_filters=num_filters
    )


@pytest.fixture
def synthesis_transform(latent_channels, num_filters):
    """Synthesis transform (decoder) instance."""
    from src.neural_compression.transforms import SynthesisTransform

    return SynthesisTransform(
        out_channels=3, latent_channels=latent_channels, num_filters=num_filters
    )


@pytest.fixture
def factorized_prior(hyper_channels):
    """Factorized prior instance."""
    from src.neural_compression.entropy import FactorizedPrior

    return FactorizedPrior(hyper_channels)


@pytest.fixture
def entropy_model(latent_channels, hyper_channels):
    """Entropy model instance."""
    from src.neural_compression.entropy import EntropyModel

    return EntropyModel(latent_channels, hyper_channels)


@pytest.fixture
def gaussian_entropy_model():
    """Gaussian entropy model instance."""
    from src.neural_compression.entropy import GaussianEntropyModel

    return GaussianEntropyModel()


@pytest.fixture
def arithmetic_coder():
    """Arithmetic coder instance."""
    from src.neural_compression.codecs import ArithmeticCoder

    return ArithmeticCoder()


@pytest.fixture
def range_coder():
    """Range coder instance."""
    from src.neural_compression.codecs import RangeCoder

    return RangeCoder()


@pytest.fixture
def neural_codec(latent_channels, hyper_channels, num_filters):
    """Neural compression codec instance."""
    from src.neural_compression.codecs import NeuralCompressionCodec

    return NeuralCompressionCodec(
        latent_channels=latent_channels,
        hyper_channels=hyper_channels,
        num_filters=num_filters,
    )


# Utility fixtures


@pytest.fixture
def tolerance():
    """Default numerical tolerance for comparisons."""
    return 1e-5


@pytest.fixture
def gradient_check_eps():
    """Epsilon for gradient checking."""
    return 1e-4
