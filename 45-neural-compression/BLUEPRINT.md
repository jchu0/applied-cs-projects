# Project 44: Neural Compression Engine (Hyperprior Codecs / DeepMind-style)

> **Concepts covered:** §03 ml-engineering — `02-deep-learning`; §04 ai-engineering — `07-custom-models`

## Executive Summary

A neural compression system implementing learned image and data compression using VAE-based architectures with hyperprior entropy models. This project focuses on rate-distortion optimization, arithmetic coding, and learned quantization to achieve compression ratios superior to traditional codecs like JPEG and WebP.

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                   Neural Compression Engine                       |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Encoder           |     | Entropy Model     |     | Arithmetic| |
|  | (CNN/Transform)   |---->| (Hyperprior)      |---->| Coder     | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Quantizer         |     | Probability       |     | Bitstream | |
|  | (Learned)         |     | Estimator         |     | Writer    | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Decoder                                |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | Arith  |  | Dequant|  | Hyper  |  | Main   |           |     |
|  |  | Decode |  | izer   |  | Decode |  | Decode |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Analysis and Synthesis Transforms

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GDN(nn.Module):
    """Generalized Divisive Normalization for compression."""

    def __init__(self, num_channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.num_channels = num_channels

        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(num_channels))
        self.gamma = nn.Parameter(torch.eye(num_channels) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        gamma = self.gamma.abs() + 1e-6
        beta = self.beta.abs() + 1e-6

        # Compute normalization factor
        # norm[i] = sqrt(beta[i] + sum_j(gamma[i,j] * x[j]^2))
        x_sq = x ** 2
        norm = beta.view(1, -1, 1, 1) + F.conv2d(
            x_sq, gamma.unsqueeze(-1).unsqueeze(-1)
        )
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm


class AnalysisTransform(nn.Module):
    """Encoder network: Image -> Latent representation."""

    def __init__(self,
                 in_channels: int = 3,
                 latent_channels: int = 192,
                 num_filters: int = 128):
        super().__init__()

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
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            [B, latent_channels, H/16, W/16] latent
        """
        return self.layers(x)


class SynthesisTransform(nn.Module):
    """Decoder network: Latent -> Reconstructed image."""

    def __init__(self,
                 out_channels: int = 3,
                 latent_channels: int = 192,
                 num_filters: int = 128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, num_filters, 5, stride=2, padding=2, output_padding=1),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(num_filters, out_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B, latent_channels, H/16, W/16] latent

        Returns:
            [B, 3, H, W] reconstructed image
        """
        return self.layers(y)
```

#### 2. Hyperprior Entropy Model

```python
class HyperAnalysis(nn.Module):
    """Hyper-encoder for entropy parameters."""

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.abs(y))


class HyperSynthesis(nn.Module):
    """Hyper-decoder for entropy parameters."""

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, latent_channels * 2, 3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and scale for Gaussian entropy model."""
        out = self.layers(z)
        mean, log_scale = out.chunk(2, dim=1)
        scale = torch.exp(log_scale)
        return mean, scale


class EntropyModel(nn.Module):
    """
    Entropy model with hyperprior.

    Estimates probability distribution of latents for arithmetic coding.
    """

    def __init__(self, latent_channels: int = 192, hyper_channels: int = 128):
        super().__init__()

        self.hyper_analysis = HyperAnalysis(latent_channels, hyper_channels)
        self.hyper_synthesis = HyperSynthesis(latent_channels, hyper_channels)

        # For quantized hyper-latents
        self.hyper_entropy = FactorizedPrior(hyper_channels)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute entropy parameters and estimate bits.

        Args:
            y: [B, C, H, W] latent representation

        Returns:
            y_hat: Quantized latent
            z_hat: Quantized hyper-latent
            likelihoods: Dictionary of likelihoods for rate computation
        """
        # Hyper-encoder
        z = self.hyper_analysis(y)

        # Quantize hyper-latent
        z_hat, z_likelihood = self.hyper_entropy(z, training=self.training)

        # Hyper-decoder
        mean, scale = self.hyper_synthesis(z_hat)

        # Quantize main latent
        if self.training:
            # Add uniform noise for differentiable approximation
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)

        # Compute likelihoods using Gaussian model
        y_likelihood = self._gaussian_likelihood(y_hat, mean, scale)

        return y_hat, z_hat, {
            'y': y_likelihood,
            'z': z_likelihood
        }

    def _gaussian_likelihood(self,
                              y: torch.Tensor,
                              mean: torch.Tensor,
                              scale: torch.Tensor) -> torch.Tensor:
        """Compute likelihood under Gaussian distribution."""
        # Integrate Gaussian over quantization bin
        half = 0.5
        upper = self._standardized_cumulative((y + half - mean) / scale)
        lower = self._standardized_cumulative((y - half - mean) / scale)
        likelihood = upper - lower
        return likelihood.clamp(min=1e-9)

    def _standardized_cumulative(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal CDF."""
        return 0.5 * (1 + torch.erf(x / (2 ** 0.5)))


class FactorizedPrior(nn.Module):
    """Factorized prior for hyper-latents."""

    def __init__(self, channels: int, init_scale: float = 10.0):
        super().__init__()
        self.channels = channels

        # Learnable CDF parameters
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.eye(channels).unsqueeze(0) * init_scale)
            for _ in range(4)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, channels, 1))
            for _ in range(4)
        ])
        self.factors = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1))
            for _ in range(4)
        ])

    def forward(self,
                z: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize and compute likelihood.

        Returns:
            z_hat: Quantized values
            likelihood: Probability of each symbol
        """
        if training:
            z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_hat = torch.round(z)

        # Compute likelihood using learned CDF
        likelihood = self._compute_likelihood(z_hat)

        return z_hat, likelihood

    def _compute_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """Compute likelihood using learned factorized prior."""
        # Simplified: use logistic distribution
        # Full implementation would use learned CDF
        scale = 1.0
        upper = torch.sigmoid((z + 0.5) / scale)
        lower = torch.sigmoid((z - 0.5) / scale)
        return (upper - lower).clamp(min=1e-9)
```

#### 3. Arithmetic Coding

```python
import numpy as np
from typing import List

class ArithmeticCoder:
    """
    Arithmetic coder for lossless entropy coding.

    Encodes symbols given their probability distributions.
    """

    def __init__(self, precision: int = 16):
        self.precision = precision
        self.max_range = 1 << precision
        self.half = self.max_range >> 1
        self.quarter = self.max_range >> 2

    def encode(self,
               symbols: np.ndarray,
               cdfs: np.ndarray) -> bytes:
        """
        Encode symbols using arithmetic coding.

        Args:
            symbols: [N] integer symbols to encode
            cdfs: [N, num_symbols+1] cumulative distribution functions

        Returns:
            Compressed bitstream
        """
        bits = []
        low = 0
        high = self.max_range
        pending_bits = 0

        for i, symbol in enumerate(symbols):
            # Get range for this symbol
            range_size = high - low
            cdf = cdfs[i]

            # Scale CDF to current range
            high = low + (range_size * cdf[symbol + 1]) // cdf[-1]
            low = low + (range_size * cdf[symbol]) // cdf[-1]

            # Emit bits
            while True:
                if high < self.half:
                    # Emit 0 followed by pending 1s
                    bits.append(0)
                    bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.half:
                    # Emit 1 followed by pending 0s
                    bits.append(1)
                    bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    # E3 scaling
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low = low << 1
                high = (high << 1) + 1

        # Finalize
        pending_bits += 1
        if low < self.quarter:
            bits.append(0)
            bits.extend([1] * pending_bits)
        else:
            bits.append(1)
            bits.extend([0] * pending_bits)

        # Convert to bytes
        return self._bits_to_bytes(bits)

    def decode(self,
               bitstream: bytes,
               cdfs: np.ndarray,
               num_symbols: int) -> np.ndarray:
        """
        Decode symbols from bitstream.

        Args:
            bitstream: Compressed data
            cdfs: [num_symbols, vocab_size+1] CDFs
            num_symbols: Number of symbols to decode

        Returns:
            [num_symbols] decoded integers
        """
        bits = self._bytes_to_bits(bitstream)
        bit_idx = 0

        # Initialize value from first precision bits
        value = 0
        for _ in range(self.precision):
            value = (value << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1

        low = 0
        high = self.max_range
        symbols = []

        for i in range(num_symbols):
            range_size = high - low
            cdf = cdfs[i]

            # Find symbol
            scaled_value = ((value - low + 1) * cdf[-1] - 1) // range_size
            symbol = np.searchsorted(cdf[1:], scaled_value, side='right')
            symbols.append(symbol)

            # Update range
            high = low + (range_size * cdf[symbol + 1]) // cdf[-1]
            low = low + (range_size * cdf[symbol]) // cdf[-1]

            # Renormalize
            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    value -= self.half
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    value -= self.quarter
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low = low << 1
                high = (high << 1) + 1
                value = (value << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
                bit_idx += 1

        return np.array(symbols)

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert bit list to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8:
            bits.append(0)

        result = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)

        return bytes(result)

    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to bit list."""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits
```

#### 4. Complete Compression Codec

```python
class NeuralCompressionCodec(nn.Module):
    """
    Complete neural compression codec.

    Implements end-to-end learned image compression with
    rate-distortion optimization.
    """

    def __init__(self,
                 latent_channels: int = 192,
                 hyper_channels: int = 128,
                 num_filters: int = 128):
        super().__init__()

        # Main autoencoder
        self.encoder = AnalysisTransform(3, latent_channels, num_filters)
        self.decoder = SynthesisTransform(3, latent_channels, num_filters)

        # Entropy model
        self.entropy_model = EntropyModel(latent_channels, hyper_channels)

        # Arithmetic coder
        self.coder = ArithmeticCoder()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for training.

        Args:
            x: [B, 3, H, W] input image

        Returns:
            x_hat: Reconstructed image
            losses: Dictionary of loss components
        """
        # Encode
        y = self.encoder(x)

        # Entropy modeling and quantization
        y_hat, z_hat, likelihoods = self.entropy_model(y)

        # Decode
        x_hat = self.decoder(y_hat)

        # Compute losses
        losses = self._compute_losses(x, x_hat, likelihoods)

        return x_hat, losses

    def _compute_losses(self,
                        x: torch.Tensor,
                        x_hat: torch.Tensor,
                        likelihoods: dict) -> dict:
        """Compute rate-distortion losses."""
        # Distortion: MSE
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)

        # Rate: bits per pixel
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

        y_bits = -torch.log2(likelihoods['y']).sum() / num_pixels
        z_bits = -torch.log2(likelihoods['z']).sum() / num_pixels
        bpp = y_bits + z_bits

        return {
            'mse': mse,
            'psnr': psnr,
            'bpp': bpp,
            'y_bpp': y_bits,
            'z_bpp': z_bits
        }

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress image to bitstream.

        Args:
            x: [1, 3, H, W] input image

        Returns:
            Compressed bitstream
        """
        with torch.no_grad():
            # Encode
            y = self.encoder(x)
            z = self.entropy_model.hyper_analysis(y)

            # Quantize
            z_hat = torch.round(z)
            y_hat = torch.round(y)

            # Get entropy parameters
            mean, scale = self.entropy_model.hyper_synthesis(z_hat)

            # Build CDFs for arithmetic coding
            z_cdfs = self._build_factorized_cdfs(z_hat)
            y_cdfs = self._build_gaussian_cdfs(y_hat, mean, scale)

            # Encode
            z_flat = z_hat.flatten().cpu().numpy().astype(np.int32)
            y_flat = y_hat.flatten().cpu().numpy().astype(np.int32)

            z_bytes = self.coder.encode(z_flat, z_cdfs)
            y_bytes = self.coder.encode(y_flat, y_cdfs)

            # Pack into bitstream
            return self._pack_bitstream(
                z_bytes, y_bytes,
                z_hat.shape, y_hat.shape
            )

    def decompress(self, bitstream: bytes) -> torch.Tensor:
        """
        Decompress bitstream to image.

        Args:
            bitstream: Compressed data

        Returns:
            [1, 3, H, W] reconstructed image
        """
        with torch.no_grad():
            # Unpack bitstream
            z_bytes, y_bytes, z_shape, y_shape = self._unpack_bitstream(bitstream)

            # Decode z
            z_cdfs = self._build_factorized_cdfs_for_decode(z_shape)
            z_flat = self.coder.decode(z_bytes, z_cdfs, np.prod(z_shape))
            z_hat = torch.tensor(z_flat).reshape(z_shape).float()

            # Get entropy parameters
            mean, scale = self.entropy_model.hyper_synthesis(z_hat)

            # Decode y
            y_cdfs = self._build_gaussian_cdfs_for_decode(y_shape, mean, scale)
            y_flat = self.coder.decode(y_bytes, y_cdfs, np.prod(y_shape))
            y_hat = torch.tensor(y_flat).reshape(y_shape).float()

            # Decode image
            x_hat = self.decoder(y_hat)

            return x_hat

    def _build_gaussian_cdfs(self,
                              y: torch.Tensor,
                              mean: torch.Tensor,
                              scale: torch.Tensor) -> np.ndarray:
        """Build CDFs for Gaussian entropy model."""
        # Simplified: discretize Gaussian CDF
        # Full implementation needs proper symbol range handling
        y_flat = y.flatten()
        mean_flat = mean.flatten()
        scale_flat = scale.flatten()

        num_symbols = len(y_flat)
        max_symbol = 256  # Quantization levels

        cdfs = np.zeros((num_symbols, max_symbol + 1), dtype=np.int32)
        for i in range(num_symbols):
            for s in range(max_symbol + 1):
                val = (s - max_symbol // 2 - mean_flat[i].item()) / scale_flat[i].item()
                cdfs[i, s] = int(self._gaussian_cdf(val) * (1 << 16))

        return cdfs

    def _gaussian_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        import math
        return 0.5 * (1 + math.erf(x / (2 ** 0.5)))

    def _build_factorized_cdfs(self, z: torch.Tensor) -> np.ndarray:
        """Build CDFs for factorized prior."""
        # Simplified implementation
        num_symbols = z.numel()
        max_symbol = 256
        cdfs = np.zeros((num_symbols, max_symbol + 1), dtype=np.int32)

        for i in range(num_symbols):
            for s in range(max_symbol + 1):
                cdfs[i, s] = int((s / max_symbol) * (1 << 16))

        return cdfs

    def _pack_bitstream(self,
                        z_bytes: bytes,
                        y_bytes: bytes,
                        z_shape: tuple,
                        y_shape: tuple) -> bytes:
        """Pack data into single bitstream with header."""
        import struct

        header = struct.pack('IIII',
            len(z_bytes),
            len(y_bytes),
            np.prod(z_shape),
            np.prod(y_shape)
        )

        # Shape info
        shape_info = struct.pack('IIII', *z_shape) + struct.pack('IIII', *y_shape)

        return header + shape_info + z_bytes + y_bytes

    def _unpack_bitstream(self, bitstream: bytes) -> Tuple:
        """Unpack bitstream."""
        import struct

        # Read header
        header_size = 4 * 4  # 4 ints
        header = struct.unpack('IIII', bitstream[:header_size])
        z_len, y_len, z_numel, y_numel = header

        # Read shapes
        shape_offset = header_size
        z_shape = struct.unpack('IIII', bitstream[shape_offset:shape_offset+16])
        y_shape = struct.unpack('IIII', bitstream[shape_offset+16:shape_offset+32])

        # Read data
        data_offset = shape_offset + 32
        z_bytes = bitstream[data_offset:data_offset+z_len]
        y_bytes = bitstream[data_offset+z_len:data_offset+z_len+y_len]

        return z_bytes, y_bytes, z_shape, y_shape
```

### Enterprise Features

#### Multi-Rate Compression

```python
class MultiRateCodec(nn.Module):
    """Codec supporting multiple rate points with single model."""

    def __init__(self, num_rates: int = 6):
        super().__init__()

        self.num_rates = num_rates

        # Shared encoder/decoder
        self.encoder = AnalysisTransform()
        self.decoder = SynthesisTransform()

        # Rate-specific gain units
        self.gains = nn.ParameterList([
            nn.Parameter(torch.ones(192) * (0.5 + 0.5 * i / num_rates))
            for i in range(num_rates)
        ])

        # Shared entropy model
        self.entropy_model = EntropyModel()

    def forward(self,
                x: torch.Tensor,
                rate_idx: int) -> Tuple[torch.Tensor, dict]:
        """Forward with specific rate point."""
        y = self.encoder(x)

        # Apply rate-specific gain
        y = y * self.gains[rate_idx].view(1, -1, 1, 1)

        y_hat, z_hat, likelihoods = self.entropy_model(y)

        # Inverse gain
        y_hat = y_hat / self.gains[rate_idx].view(1, -1, 1, 1)

        x_hat = self.decoder(y_hat)

        losses = self._compute_losses(x, x_hat, likelihoods)
        return x_hat, losses


class ScaleAdaptiveCodec(nn.Module):
    """Codec adapting to different input scales/resolutions."""

    def __init__(self):
        super().__init__()
        self.base_codec = NeuralCompressionCodec()

        # Scale-specific adapters
        self.scale_adapters = nn.ModuleDict({
            '1x': nn.Identity(),
            '2x': nn.Sequential(
                nn.Conv2d(192, 192, 1),
                nn.ReLU()
            ),
            '4x': nn.Sequential(
                nn.Conv2d(192, 192, 1),
                nn.ReLU(),
                nn.Conv2d(192, 192, 1)
            )
        })

    def forward(self, x: torch.Tensor, scale: str = '1x'):
        y = self.base_codec.encoder(x)
        y = self.scale_adapters[scale](y)
        # ... rest of pipeline
        pass
```

#### LLM Weight Compression

```python
class ModelWeightCompressor:
    """Compress neural network weights using learned compression."""

    def __init__(self, codec: NeuralCompressionCodec):
        self.codec = codec

    def compress_layer(self, weight: torch.Tensor) -> bytes:
        """Compress a weight tensor."""
        # Reshape to image-like format
        if weight.dim() == 2:
            # Linear layer: [out, in] -> [1, 1, out, in]
            w = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 4:
            # Conv layer: [out, in, h, w] -> [out, in, h, w]
            w = weight.permute(0, 1, 2, 3)
        else:
            raise ValueError(f"Unsupported weight shape: {weight.shape}")

        # Normalize to [0, 1] range
        w_min, w_max = w.min(), w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)

        # Compress
        bitstream = self.codec.compress(w_norm.unsqueeze(0) if w.dim() == 4 else w)

        # Add metadata
        import struct
        meta = struct.pack('ff', w_min.item(), w_max.item())

        return meta + bitstream

    def decompress_layer(self, bitstream: bytes, shape: tuple) -> torch.Tensor:
        """Decompress weight tensor."""
        import struct

        # Extract metadata
        w_min, w_max = struct.unpack('ff', bitstream[:8])
        data = bitstream[8:]

        # Decompress
        w_norm = self.codec.decompress(data)

        # Denormalize
        w = w_norm * (w_max - w_min) + w_min

        # Reshape back
        return w.squeeze().reshape(shape)
```

## API Reference

### Compression

```python
# Create codec
codec = NeuralCompressionCodec(latent_channels=192)

# Load pretrained weights
codec.load_state_dict(torch.load('codec.pth'))

# Compress image
image = load_image('photo.png')  # [1, 3, H, W]
bitstream = codec.compress(image)

# Check size
original_size = image.numel() * 4  # float32
compressed_size = len(bitstream)
ratio = original_size / compressed_size
```

### Decompression

```python
# Decompress
reconstructed = codec.decompress(bitstream)

# Save
save_image(reconstructed, 'reconstructed.png')
```

### Training

```python
# Rate-distortion loss
lambda_rd = 0.01  # Rate-distortion tradeoff

for batch in dataloader:
    x_hat, losses = codec(batch)

    loss = losses['mse'] + lambda_rd * losses['bpp']
    loss.backward()
    optimizer.step()
```

## Implementation Phases

### Phase 1: Transforms (Weeks 1-2)
- Analysis transform (encoder)
- Synthesis transform (decoder)
- GDN normalization
- Basic autoencoder training

### Phase 2: Entropy Model (Weeks 3-4)
- Factorized prior
- Hyperprior network
- Gaussian entropy model
- Likelihood computation

### Phase 3: Arithmetic Coding (Weeks 5-6)
- Arithmetic encoder
- Arithmetic decoder
- CDF construction
- Bitstream packing

### Phase 4: Training (Weeks 7-8)
- Rate-distortion loss
- Multi-rate training
- Perceptual losses (LPIPS)
- Training stability

### Phase 5: Optimization (Weeks 9-10)
- Decoder optimization
- Quantized inference
- Fast entropy coding

### Phase 6: Enterprise (Weeks 11-14)
- Multi-rate codec
- Model weight compression
- Edge deployment

## Testing Strategy

### Unit Tests

```python
class TestTransforms:
    def test_encoder_decoder(self):
        encoder = AnalysisTransform()
        decoder = SynthesisTransform()

        x = torch.randn(1, 3, 256, 256)
        y = encoder(x)
        x_hat = decoder(y)

        assert x_hat.shape == x.shape

class TestArithmeticCoding:
    def test_encode_decode(self):
        coder = ArithmeticCoder()

        symbols = np.array([0, 1, 2, 1, 0])
        cdfs = np.array([[0, 100, 200, 300]] * 5)

        encoded = coder.encode(symbols, cdfs)
        decoded = coder.decode(encoded, cdfs, 5)

        assert np.array_equal(symbols, decoded)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| PSNR | >35 dB | At 0.5 bpp |
| Compression ratio | >50x | vs PNG |
| Encode speed | >10 fps | 1080p |
| Decode speed | >30 fps | 1080p |

## Dependencies

- PyTorch >= 2.0
- NumPy
- Pillow (for image I/O)
- (Optional) CompressAI for comparison

## References

- Variational Image Compression with a Scale Hyperprior
- Joint Autoregressive and Hierarchical Priors for Learned Image Compression
- Practical Full Resolution Learned Lossless Image Compression
