"""Codecs for Neural Compression.

This module implements:
- ArithmeticCoder: Arithmetic coding for lossless compression
- RangeCoder: Alternative range-based coder
- NeuralCompressionCodec: Full end-to-end compression codec
"""

import math
import struct
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import AnalysisTransform, SynthesisTransform
from .entropy import EntropyModel


@dataclass
class CompressionResult:
    """Result of compression operation."""

    bitstream: bytes
    original_size: int  # in bytes
    compressed_size: int  # in bytes
    compression_ratio: float
    bpp: float  # bits per pixel
    shape: Tuple[int, ...]


class ArithmeticCoder:
    """Arithmetic coder for lossless entropy coding.

    Encodes symbols given their probability distributions using
    arithmetic coding with E1/E2/E3 scaling for precision management.

    Args:
        precision: Number of bits for the range (default 16)
    """

    def __init__(self, precision: int = 16):
        self.precision = precision
        self.max_range = 1 << precision
        self.half = self.max_range >> 1
        self.quarter = self.max_range >> 2
        self.three_quarter = 3 * self.quarter

    def encode(self, symbols: np.ndarray, cdfs: np.ndarray) -> bytes:
        """Encode symbols using arithmetic coding.

        Args:
            symbols: [N] integer symbols to encode (0 to vocab_size-1)
            cdfs: [N, vocab_size+1] cumulative distribution functions
                  CDFs should be scaled to [0, 2^precision]

        Returns:
            Compressed bitstream
        """
        bits: List[int] = []
        low = 0
        high = self.max_range

        pending_bits = 0

        for i, symbol in enumerate(symbols):
            symbol = int(symbol)
            range_size = high - low
            cdf = cdfs[i]

            # Ensure CDF is properly normalized
            total = int(cdf[-1])
            if total == 0:
                total = 1

            # Update range based on symbol
            low_cdf = int(cdf[symbol])
            high_cdf = int(cdf[symbol + 1])

            high = low + (range_size * high_cdf) // total
            low = low + (range_size * low_cdf) // total

            # Emit bits and handle underflow
            while True:
                if high < self.half:
                    # E1: Emit 0, followed by pending 1s
                    bits.append(0)
                    bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.half:
                    # E2: Emit 1, followed by pending 0s
                    bits.append(1)
                    bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarter:
                    # E3: Underflow, track pending bits
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                # Scale up
                low = low << 1
                high = (high << 1) + 1

        # Finalize stream
        pending_bits += 1
        if low < self.quarter:
            bits.append(0)
            bits.extend([1] * pending_bits)
        else:
            bits.append(1)
            bits.extend([0] * pending_bits)

        return self._bits_to_bytes(bits)

    def decode(
        self, bitstream: bytes, cdfs: np.ndarray, num_symbols: int
    ) -> np.ndarray:
        """Decode symbols from bitstream.

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
            if bit_idx < len(bits):
                value = (value << 1) | bits[bit_idx]
                bit_idx += 1
            else:
                value = value << 1

        low = 0
        high = self.max_range
        symbols = []

        for i in range(num_symbols):
            range_size = high - low
            cdf = cdfs[i]
            total = int(cdf[-1])
            if total == 0:
                total = 1

            # Find symbol by binary search on CDF
            scaled_value = ((value - low + 1) * total - 1) // range_size

            # Binary search for symbol
            symbol = 0
            for s in range(len(cdf) - 1):
                if cdf[s] <= scaled_value < cdf[s + 1]:
                    symbol = s
                    break

            symbols.append(symbol)

            # Update range
            low_cdf = int(cdf[symbol])
            high_cdf = int(cdf[symbol + 1])
            high = low + (range_size * high_cdf) // total
            low = low + (range_size * low_cdf) // total

            # Renormalize
            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    value -= self.half
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarter:
                    value -= self.quarter
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low = low << 1
                high = (high << 1) + 1
                if bit_idx < len(bits):
                    value = (value << 1) | bits[bit_idx]
                    bit_idx += 1
                else:
                    value = value << 1

        return np.array(symbols, dtype=np.int32)

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


class RangeCoder:
    """Range coder - alternative to arithmetic coding.

    Uses range-based coding which is slightly faster than
    arithmetic coding with similar compression efficiency.

    Args:
        precision: Number of bits for range (default 32)
    """

    def __init__(self, precision: int = 32):
        self.precision = precision
        self.top = 1 << (precision - 8)
        self.bottom = 1 << (precision - 16)
        self.max_range = 1 << precision

    def encode(
        self, symbols: np.ndarray, probs: np.ndarray, vocab_size: int = 256
    ) -> bytes:
        """Encode symbols using range coding.

        Args:
            symbols: [N] integer symbols
            probs: [N, vocab_size] probability for each symbol position
            vocab_size: Size of symbol vocabulary

        Returns:
            Compressed bytes
        """
        output = []
        low = 0
        range_ = self.max_range

        for i, symbol in enumerate(symbols):
            symbol = int(symbol)
            prob = probs[i]

            # Build CDF from probabilities
            cdf = np.zeros(vocab_size + 1, dtype=np.float64)
            cdf[1:] = np.cumsum(prob)
            cdf = (cdf / cdf[-1] * (1 << 16)).astype(np.int64)

            # Update range
            total = int(cdf[-1])
            low_p = int(cdf[symbol])
            high_p = int(cdf[symbol + 1])

            range_ = range_ // total
            low = low + range_ * low_p
            range_ = range_ * (high_p - low_p)

            # Renormalize
            while range_ < self.bottom:
                output.append((low >> (self.precision - 8)) & 0xFF)
                low = (low << 8) & (self.max_range - 1)
                range_ = range_ << 8

        # Flush remaining bytes
        for _ in range(4):
            output.append((low >> (self.precision - 8)) & 0xFF)
            low = (low << 8) & (self.max_range - 1)

        return bytes(output)

    def decode(
        self,
        bitstream: bytes,
        probs: np.ndarray,
        num_symbols: int,
        vocab_size: int = 256,
    ) -> np.ndarray:
        """Decode symbols from bitstream.

        Args:
            bitstream: Compressed data
            probs: [num_symbols, vocab_size] probabilities
            num_symbols: Number of symbols to decode
            vocab_size: Size of vocabulary

        Returns:
            Decoded symbols
        """
        data = list(bitstream)
        byte_idx = 0

        # Initialize code value
        code = 0
        for _ in range(4):
            if byte_idx < len(data):
                code = (code << 8) | data[byte_idx]
                byte_idx += 1
            else:
                code = code << 8

        low = 0
        range_ = self.max_range
        symbols = []

        for i in range(num_symbols):
            prob = probs[i]

            # Build CDF
            cdf = np.zeros(vocab_size + 1, dtype=np.float64)
            cdf[1:] = np.cumsum(prob)
            cdf = (cdf / cdf[-1] * (1 << 16)).astype(np.int64)

            total = int(cdf[-1])
            range_ = range_ // total
            offset = (code - low) // range_

            # Find symbol
            symbol = 0
            for s in range(vocab_size):
                if cdf[s] <= offset < cdf[s + 1]:
                    symbol = s
                    break

            symbols.append(symbol)

            # Update range
            low = low + range_ * int(cdf[symbol])
            range_ = range_ * (int(cdf[symbol + 1]) - int(cdf[symbol]))

            # Renormalize
            while range_ < self.bottom:
                if byte_idx < len(data):
                    code = ((code << 8) | data[byte_idx]) & (self.max_range - 1)
                    byte_idx += 1
                else:
                    code = (code << 8) & (self.max_range - 1)
                low = (low << 8) & (self.max_range - 1)
                range_ = range_ << 8

        return np.array(symbols, dtype=np.int32)


class NeuralCompressionCodec(nn.Module):
    """Complete neural compression codec.

    Implements end-to-end learned image compression with
    rate-distortion optimization.

    Args:
        latent_channels: Number of latent channels
        hyper_channels: Number of hyper-latent channels
        num_filters: Number of intermediate filters
    """

    def __init__(
        self,
        latent_channels: int = 192,
        hyper_channels: int = 128,
        num_filters: int = 128,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.num_filters = num_filters

        # Main autoencoder
        self.encoder = AnalysisTransform(3, latent_channels, num_filters)
        self.decoder = SynthesisTransform(3, latent_channels, num_filters)

        # Entropy model
        self.entropy_model = EntropyModel(latent_channels, hyper_channels)

        # Arithmetic coder
        self.coder = ArithmeticCoder()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for training.

        Args:
            x: Input image [B, 3, H, W] in range [0, 1]

        Returns:
            x_hat: Reconstructed image
            likelihoods: Dictionary with 'y' and 'z' likelihood tensors
        """
        # Encode
        y = self.encoder(x)

        # Entropy modeling and quantization
        y_hat, z_hat, likelihoods = self.entropy_model(y)

        # Decode
        x_hat = self.decoder(y_hat)

        # Return likelihoods dict for RateDistortionLoss compatibility
        return x_hat, likelihoods

    def _compute_losses(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        likelihoods: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute rate-distortion losses.

        Args:
            x: Original image
            x_hat: Reconstructed image
            likelihoods: Dictionary with y and z likelihoods

        Returns:
            Dictionary with loss components
        """
        # Distortion: MSE
        mse = F.mse_loss(x_hat, x)

        # PSNR (for logging)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        # Rate: bits per pixel
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
        }

    def compress(self, x: torch.Tensor) -> CompressionResult:
        """Compress image to bitstream.

        Args:
            x: Input image [1, 3, H, W]

        Returns:
            CompressionResult with bitstream and metadata
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("Input must be [1, 3, H, W]")

        original_shape = x.shape
        original_size = x.numel() * 4  # float32 bytes

        with torch.no_grad():
            # Encode
            y = self.encoder(x)
            y_hat, z_hat, mean, scale = self.entropy_model.compress(y)

            # Build CDFs for arithmetic coding
            z_cdfs = self._build_factorized_cdfs(z_hat)
            y_cdfs = self._build_gaussian_cdfs(y_hat, mean, scale)

            # Flatten and shift to non-negative range
            z_flat = z_hat.flatten().cpu().numpy().astype(np.int32)
            y_flat = y_hat.flatten().cpu().numpy().astype(np.int32)

            # Shift to positive symbols
            z_offset = 128
            y_offset = 128
            z_symbols = z_flat + z_offset
            y_symbols = y_flat + y_offset

            # Clamp to valid range
            z_symbols = np.clip(z_symbols, 0, 255)
            y_symbols = np.clip(y_symbols, 0, 255)

            # Encode
            z_bytes = self.coder.encode(z_symbols, z_cdfs)
            y_bytes = self.coder.encode(y_symbols, y_cdfs)

            # Pack into bitstream
            bitstream = self._pack_bitstream(
                z_bytes, y_bytes, z_hat.shape, y_hat.shape
            )

        compressed_size = len(bitstream)
        num_pixels = x.shape[2] * x.shape[3]
        bpp = (compressed_size * 8) / num_pixels

        return CompressionResult(
            bitstream=bitstream,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            bpp=bpp,
            shape=original_shape,
        )

    def decompress(
        self, bitstream: bytes, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Decompress bitstream to image.

        Args:
            bitstream: Compressed data
            device: Device to put output tensor on

        Returns:
            Reconstructed image [1, 3, H, W]
        """
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            # Unpack bitstream
            z_bytes, y_bytes, z_shape, y_shape = self._unpack_bitstream(bitstream)

            # Build CDFs for hyper-latent
            z_cdfs = self._build_factorized_cdfs_for_decode(z_shape)

            # Decode z
            z_numel = int(np.prod(z_shape))
            z_symbols = self.coder.decode(z_bytes, z_cdfs, z_numel)
            z_flat = z_symbols.astype(np.float32) - 128
            z_hat = torch.tensor(z_flat, device=device).reshape(z_shape)

            # Get entropy parameters
            mean, scale = self.entropy_model.get_entropy_parameters(z_hat)

            # Build CDFs for main latent
            y_cdfs = self._build_gaussian_cdfs_for_decode(y_shape, mean, scale)

            # Decode y
            y_numel = int(np.prod(y_shape))
            y_symbols = self.coder.decode(y_bytes, y_cdfs, y_numel)
            y_flat = y_symbols.astype(np.float32) - 128
            y_hat = torch.tensor(y_flat, device=device).reshape(y_shape)

            # Decode image
            x_hat = self.decoder(y_hat)

        return x_hat

    def _build_gaussian_cdfs(
        self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
    ) -> np.ndarray:
        """Build CDFs for Gaussian entropy model."""
        y_flat = y.flatten().cpu()
        mean_flat = mean.flatten().cpu()
        scale_flat = scale.flatten().cpu()

        num_symbols = len(y_flat)
        max_symbol = 256
        precision = 16

        cdfs = np.zeros((num_symbols, max_symbol + 1), dtype=np.int64)
        for i in range(num_symbols):
            m = mean_flat[i].item()
            s = max(scale_flat[i].item(), 0.1)

            for sym in range(max_symbol + 1):
                # CDF at symbol boundary (shifted by 128)
                val = (sym - 128 - 0.5 - m) / s
                cdf_val = 0.5 * (1 + math.erf(val / math.sqrt(2)))
                cdfs[i, sym] = int(cdf_val * (1 << precision))

            # Ensure monotonicity
            for sym in range(1, max_symbol + 1):
                if cdfs[i, sym] <= cdfs[i, sym - 1]:
                    cdfs[i, sym] = cdfs[i, sym - 1] + 1

        return cdfs

    def _build_factorized_cdfs(self, z: torch.Tensor) -> np.ndarray:
        """Build CDFs for factorized prior."""
        num_symbols = z.numel()
        max_symbol = 256
        precision = 16

        # Use uniform-like distribution as simplified factorized prior
        cdfs = np.zeros((num_symbols, max_symbol + 1), dtype=np.int64)
        for i in range(num_symbols):
            for sym in range(max_symbol + 1):
                # Simple logistic CDF centered at 128
                val = (sym - 128) / 10.0
                cdf_val = 1.0 / (1.0 + math.exp(-val))
                cdfs[i, sym] = int(cdf_val * (1 << precision))

            # Ensure monotonicity
            for sym in range(1, max_symbol + 1):
                if cdfs[i, sym] <= cdfs[i, sym - 1]:
                    cdfs[i, sym] = cdfs[i, sym - 1] + 1

        return cdfs

    def _build_factorized_cdfs_for_decode(self, z_shape: Tuple) -> np.ndarray:
        """Build CDFs for decoding hyper-latent."""
        return self._build_factorized_cdfs(torch.zeros(z_shape))

    def _build_gaussian_cdfs_for_decode(
        self, y_shape: Tuple, mean: torch.Tensor, scale: torch.Tensor
    ) -> np.ndarray:
        """Build CDFs for decoding main latent."""
        return self._build_gaussian_cdfs(torch.zeros(y_shape), mean, scale)

    def _pack_bitstream(
        self,
        z_bytes: bytes,
        y_bytes: bytes,
        z_shape: Tuple,
        y_shape: Tuple,
    ) -> bytes:
        """Pack data into single bitstream with header."""
        # Header format:
        # - 4 bytes: z_bytes length
        # - 4 bytes: y_bytes length
        # - 16 bytes: z_shape (4 x 4 bytes)
        # - 16 bytes: y_shape (4 x 4 bytes)
        # - z_bytes
        # - y_bytes

        header = struct.pack("II", len(z_bytes), len(y_bytes))

        # Pack shapes (pad to 4 elements)
        z_shape_padded = list(z_shape) + [1] * (4 - len(z_shape))
        y_shape_padded = list(y_shape) + [1] * (4 - len(y_shape))

        shape_info = struct.pack("IIII", *z_shape_padded[:4])
        shape_info += struct.pack("IIII", *y_shape_padded[:4])

        return header + shape_info + z_bytes + y_bytes

    def _unpack_bitstream(self, bitstream: bytes) -> Tuple:
        """Unpack bitstream."""
        # Read header
        header_size = 8  # 2 x 4 bytes
        z_len, y_len = struct.unpack("II", bitstream[:header_size])

        # Read shapes
        shape_offset = header_size
        z_shape = struct.unpack("IIII", bitstream[shape_offset : shape_offset + 16])
        y_shape = struct.unpack(
            "IIII", bitstream[shape_offset + 16 : shape_offset + 32]
        )

        # Read data
        data_offset = shape_offset + 32
        z_bytes = bitstream[data_offset : data_offset + z_len]
        y_bytes = bitstream[data_offset + z_len : data_offset + z_len + y_len]

        return z_bytes, y_bytes, z_shape, y_shape

    def training_step(
        self, x: torch.Tensor, lambda_rd: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single training step.

        Args:
            x: Input batch [B, 3, H, W]
            lambda_rd: Rate-distortion tradeoff parameter

        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        x_hat, likelihoods = self.forward(x)

        # Compute losses from likelihoods
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        # Rate: bits per pixel
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        y_bpp = -torch.log2(likelihoods["y"].clamp(min=1e-9)).sum() / num_pixels
        z_bpp = -torch.log2(likelihoods["z"].clamp(min=1e-9)).sum() / num_pixels
        bpp = y_bpp + z_bpp

        # Rate-distortion loss
        loss = mse + lambda_rd * bpp

        metrics = {
            "loss": loss,
            "mse": mse,
            "psnr": psnr,
            "bpp": bpp,
        }

        return loss, metrics
