"""Neural Compression Engine - Hyperprior Codecs / DeepMind-style compression."""

from .transforms import GDN, AnalysisTransform, SynthesisTransform
from .entropy import (
    FactorizedPrior,
    HyperAnalysis,
    HyperSynthesis,
    EntropyModel,
    GaussianEntropyModel,
    MeanScaleHyperprior,
)
from .codecs import (
    ArithmeticCoder,
    RangeCoder,
    NeuralCompressionCodec,
    CompressionResult,
)
from .losses import (
    SSIMLoss,
    MS_SSIMLoss,
    LPIPSLoss,
    VGGPerceptualLoss,
    RateDistortionLoss,
    CharbonnierLoss,
)
from .training import (
    CompressionTrainer,
    TrainingConfig,
    TrainingState,
    MultiRateTrain,
)
from .data import (
    ImageFolderDataset,
    RandomCropTransform,
    CenterCropTransform,
    ComposeTransform,
    KodakDataset,
    create_dataloaders,
)
from .multirate import (
    MultiRateCodec,
    GainUnit,
    ScaleHyperpriorCodec,
)

__version__ = "0.1.0"

__all__ = [
    # Transforms
    "GDN",
    "AnalysisTransform",
    "SynthesisTransform",
    # Entropy Models
    "FactorizedPrior",
    "HyperAnalysis",
    "HyperSynthesis",
    "EntropyModel",
    "GaussianEntropyModel",
    "MeanScaleHyperprior",
    # Codecs
    "ArithmeticCoder",
    "RangeCoder",
    "NeuralCompressionCodec",
    "CompressionResult",
    # Losses
    "SSIMLoss",
    "MS_SSIMLoss",
    "LPIPSLoss",
    "VGGPerceptualLoss",
    "RateDistortionLoss",
    "CharbonnierLoss",
    # Training
    "CompressionTrainer",
    "TrainingConfig",
    "TrainingState",
    "MultiRateTrain",
    # Data
    "ImageFolderDataset",
    "RandomCropTransform",
    "CenterCropTransform",
    "ComposeTransform",
    "KodakDataset",
    "create_dataloaders",
    # Multi-rate
    "MultiRateCodec",
    "GainUnit",
    "ScaleHyperpriorCodec",
]
