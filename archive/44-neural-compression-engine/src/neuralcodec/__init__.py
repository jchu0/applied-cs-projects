"""NeuralCodec - Neural image compression engine."""

from .core import (
    CompressedData,
    CompressionMetrics,
    quantize,
    calculate_psnr,
    calculate_ms_ssim,
    GDN,
)
from .entropy import (
    GaussianModel,
    GMMModel,
    LaplacianModel,
    FullyFactorized,
)
from .models import (
    Hyperprior,
    MeanScaleHyperprior,
    JointAutoregressive,
    Cheng2020,
    train_step,
    evaluate,
)
from .coding import (
    RangeCoder,
    ANSCoder,
    compress_latent,
    decompress_latent,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "CompressedData",
    "CompressionMetrics",
    "quantize",
    "calculate_psnr",
    "calculate_ms_ssim",
    "GDN",
    # Entropy
    "GaussianModel",
    "GMMModel",
    "LaplacianModel",
    "FullyFactorized",
    # Models
    "Hyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressive",
    "Cheng2020",
    "train_step",
    "evaluate",
    # Coding
    "RangeCoder",
    "ANSCoder",
    "compress_latent",
    "decompress_latent",
]
