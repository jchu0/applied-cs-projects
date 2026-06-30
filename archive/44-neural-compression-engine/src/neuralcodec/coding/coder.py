"""Entropy coding utilities for neural compression."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RangeCoder:
    """
    Range coder for entropy coding.

    Near-optimal compression to entropy bound.
    """

    def __init__(self, precision: int = 16):
        self.precision = precision
        self.whole = 1 << precision
        self.half = self.whole >> 1
        self.quarter = self.whole >> 2

    def encode(
        self,
        symbols: np.ndarray,
        cdf: np.ndarray
    ) -> bytes:
        """
        Encode symbols using CDF.

        Args:
            symbols: Integer symbols to encode
            cdf: Cumulative distribution (num_symbols+1,)

        Returns:
            Encoded bytes
        """
        symbols = symbols.flatten().astype(np.int32)

        # Initialize
        low = 0
        high = self.whole

        bits = []

        for sym in symbols:
            # Get range
            range_size = high - low
            low = low + (range_size * cdf[sym]) // self.whole
            high = low + (range_size * (cdf[sym + 1] - cdf[sym])) // self.whole

            # Renormalize
            while True:
                if high <= self.half:
                    bits.append(0)
                    low = 2 * low
                    high = 2 * high
                elif low >= self.half:
                    bits.append(1)
                    low = 2 * (low - self.half)
                    high = 2 * (high - self.half)
                elif low >= self.quarter and high <= 3 * self.quarter:
                    low = 2 * (low - self.quarter)
                    high = 2 * (high - self.quarter)
                    # Pending bit
                else:
                    break

        # Finalize
        bits.append(0 if low < self.quarter else 1)

        # Pack to bytes
        byte_array = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= bits[i + j] << (7 - j)
            byte_array.append(byte)

        return bytes(byte_array)

    def decode(
        self,
        data: bytes,
        cdf: np.ndarray,
        num_symbols: int
    ) -> np.ndarray:
        """
        Decode symbols from bytes.

        Args:
            data: Encoded bytes
            cdf: Cumulative distribution
            num_symbols: Number of symbols to decode

        Returns:
            Decoded symbols
        """
        # Unpack bits
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)

        # Initialize
        low = 0
        high = self.whole
        value = 0

        for i in range(self.precision):
            if i < len(bits):
                value = (value << 1) | bits[i]

        symbols = []
        bit_idx = self.precision

        for _ in range(num_symbols):
            range_size = high - low
            scaled_value = ((value - low + 1) * self.whole - 1) // range_size

            # Find symbol
            sym = np.searchsorted(cdf[1:], scaled_value)
            symbols.append(sym)

            # Update range
            low_new = low + (range_size * cdf[sym]) // self.whole
            high = low + (range_size * cdf[sym + 1]) // self.whole
            low = low_new

            # Renormalize
            while True:
                if high <= self.half:
                    low = 2 * low
                    high = 2 * high
                    value = 2 * value
                    if bit_idx < len(bits):
                        value |= bits[bit_idx]
                        bit_idx += 1
                elif low >= self.half:
                    low = 2 * (low - self.half)
                    high = 2 * (high - self.half)
                    value = 2 * (value - self.half)
                    if bit_idx < len(bits):
                        value |= bits[bit_idx]
                        bit_idx += 1
                elif low >= self.quarter and high <= 3 * self.quarter:
                    low = 2 * (low - self.quarter)
                    high = 2 * (high - self.quarter)
                    value = 2 * (value - self.quarter)
                    if bit_idx < len(bits):
                        value |= bits[bit_idx]
                        bit_idx += 1
                else:
                    break

        return np.array(symbols)


class ArithmeticCoder:
    """
    Arithmetic coder for adaptive probability models.

    Supports per-symbol probability updates.
    """

    def __init__(self, precision: int = 32):
        self.precision = precision
        self.max_range = 1 << precision

    def encode(
        self,
        symbols: np.ndarray,
        get_prob_fn
    ) -> bytes:
        """
        Encode with adaptive probability model.

        Args:
            symbols: Symbols to encode
            get_prob_fn: Function (symbol, context) -> (low_prob, high_prob)

        Returns:
            Encoded bytes
        """
        low = 0
        high = self.max_range

        for i, sym in enumerate(symbols.flatten()):
            range_size = high - low

            # Get probability bounds
            sym_low, sym_high = get_prob_fn(sym, symbols[:i])

            # Update interval
            high = low + int(range_size * sym_high)
            low = low + int(range_size * sym_low)

        # Output middle of final interval
        mid = (low + high) // 2
        return mid.to_bytes(self.precision // 8, 'big')

    def decode(
        self,
        data: bytes,
        get_prob_fn,
        get_symbol_fn,
        num_symbols: int
    ) -> np.ndarray:
        """
        Decode with adaptive probability model.

        Args:
            data: Encoded bytes
            get_prob_fn: Probability function
            get_symbol_fn: Function to find symbol from value
            num_symbols: Number of symbols to decode

        Returns:
            Decoded symbols
        """
        value = int.from_bytes(data, 'big')

        low = 0
        high = self.max_range
        symbols = []

        for _ in range(num_symbols):
            range_size = high - low
            scaled = (value - low) / range_size

            # Find symbol
            sym = get_symbol_fn(scaled, symbols)
            symbols.append(sym)

            # Update interval
            sym_low, sym_high = get_prob_fn(sym, symbols[:-1])
            high = low + int(range_size * sym_high)
            low = low + int(range_size * sym_low)

        return np.array(symbols)


class ANSCoder:
    """
    Asymmetric Numeral Systems coder.

    Stack-based coding for fast decoding.
    """

    def __init__(self, precision: int = 16):
        self.precision = precision
        self.L = 1 << precision

    def encode(
        self,
        symbols: np.ndarray,
        frequencies: np.ndarray
    ) -> bytes:
        """
        Encode using rANS.

        Args:
            symbols: Symbols to encode
            frequencies: Symbol frequencies

        Returns:
            Encoded bytes
        """
        # Normalize frequencies to sum to L
        freqs = (frequencies / frequencies.sum() * self.L).astype(np.int64)
        freqs = np.maximum(freqs, 1)

        # Build CDF
        cdf = np.concatenate([[0], np.cumsum(freqs)])

        # Encode in reverse order
        state = self.L
        output = []

        for sym in reversed(symbols.flatten()):
            # Renormalize
            while state >= self.L * freqs[sym]:
                output.append(state & 0xFF)
                state >>= 8

            # Encode
            state = (state // freqs[sym]) * self.L + cdf[sym] + (state % freqs[sym])

        # Output final state
        while state > 0:
            output.append(state & 0xFF)
            state >>= 8

        return bytes(reversed(output))

    def decode(
        self,
        data: bytes,
        frequencies: np.ndarray,
        num_symbols: int
    ) -> np.ndarray:
        """
        Decode using rANS.

        Args:
            data: Encoded bytes
            frequencies: Symbol frequencies
            num_symbols: Number of symbols to decode

        Returns:
            Decoded symbols
        """
        # Normalize frequencies
        freqs = (frequencies / frequencies.sum() * self.L).astype(np.int64)
        freqs = np.maximum(freqs, 1)

        # Build CDF and inverse
        cdf = np.concatenate([[0], np.cumsum(freqs)])

        # Build symbol lookup
        lookup = np.zeros(self.L, dtype=np.int32)
        for sym in range(len(freqs)):
            lookup[cdf[sym]:cdf[sym + 1]] = sym

        # Read state
        state = 0
        data_idx = 0
        while data_idx < min(8, len(data)):
            state = (state << 8) | data[data_idx]
            data_idx += 1

        symbols = []
        for _ in range(num_symbols):
            # Decode
            slot = state % self.L
            sym = lookup[slot]
            symbols.append(sym)

            # Update state
            state = freqs[sym] * (state // self.L) + slot - cdf[sym]

            # Renormalize
            while state < self.L and data_idx < len(data):
                state = (state << 8) | data[data_idx]
                data_idx += 1

        return np.array(symbols)


def build_cdf(probabilities: np.ndarray, precision: int = 16) -> np.ndarray:
    """
    Build CDF from probabilities.

    Args:
        probabilities: Symbol probabilities
        precision: CDF precision in bits

    Returns:
        CDF values
    """
    scale = 1 << precision

    # Normalize and scale
    probs = probabilities / probabilities.sum()
    scaled = (probs * scale).astype(np.int64)

    # Ensure minimum value
    scaled = np.maximum(scaled, 1)

    # Adjust to sum to scale
    diff = scale - scaled.sum()
    scaled[np.argmax(scaled)] += diff

    # Build CDF
    cdf = np.concatenate([[0], np.cumsum(scaled)])

    return cdf


def compress_latent(
    latent: np.ndarray,
    entropy_model,
    coder_type: str = 'range'
) -> bytes:
    """
    Compress latent representation.

    Args:
        latent: Quantized latent values
        entropy_model: Entropy model for probabilities
        coder_type: Type of entropy coder

    Returns:
        Compressed bytes
    """
    # Get likelihoods
    likelihood = entropy_model.likelihood(latent)

    # Build CDF from likelihoods
    # This is simplified - real impl would do this per-element
    probs = likelihood.flatten()
    cdf = build_cdf(np.ones(256) / 256)  # Uniform for simplicity

    # Shift and offset for indexing
    symbols = latent.flatten().astype(np.int32) + 128  # Shift to positive

    if coder_type == 'range':
        coder = RangeCoder()
        return coder.encode(symbols, cdf)
    elif coder_type == 'ans':
        coder = ANSCoder()
        freqs = np.ones(256)
        return coder.encode(symbols, freqs)
    else:
        raise ValueError(f"Unknown coder type: {coder_type}")


def decompress_latent(
    data: bytes,
    shape: Tuple[int, ...],
    entropy_model,
    coder_type: str = 'range'
) -> np.ndarray:
    """
    Decompress latent representation.

    Args:
        data: Compressed bytes
        shape: Output shape
        entropy_model: Entropy model
        coder_type: Type of entropy coder

    Returns:
        Decompressed latent
    """
    num_symbols = np.prod(shape)
    cdf = build_cdf(np.ones(256) / 256)

    if coder_type == 'range':
        coder = RangeCoder()
        symbols = coder.decode(data, cdf, int(num_symbols))
    elif coder_type == 'ans':
        coder = ANSCoder()
        freqs = np.ones(256)
        symbols = coder.decode(data, freqs, int(num_symbols))
    else:
        raise ValueError(f"Unknown coder type: {coder_type}")

    # Shift back to original range
    latent = (symbols - 128).astype(np.float32)

    return latent.reshape(shape)
