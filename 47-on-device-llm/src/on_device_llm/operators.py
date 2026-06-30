"""SIMD-optimized operators for on-device LLM inference.

This module provides optimized implementations of core tensor operations
used in transformer inference. When Numba is available, SIMD-optimized
JIT-compiled versions are used; otherwise, NumPy fallbacks are provided.
"""

import numpy as np
from typing import Optional

# Try to import Numba for SIMD optimization
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def matmul_f32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        SIMD-optimized matrix multiplication.

        Uses tiled multiplication for cache efficiency with parallel execution.

        Args:
            A: First matrix [M, K]
            B: Second matrix [K, N]

        Returns:
            Result matrix [M, N]
        """
        M, K = A.shape
        K_, N = B.shape
        assert K == K_, "Matrix dimensions must match for multiplication"

        C = np.zeros((M, N), dtype=np.float32)

        # Tiled multiplication for cache efficiency
        tile_size = 32

        for i in prange(0, M, tile_size):
            for j in range(0, N, tile_size):
                for k in range(0, K, tile_size):
                    # Tile boundaries
                    i_end = min(i + tile_size, M)
                    j_end = min(j + tile_size, N)
                    k_end = min(k + tile_size, K)

                    # Compute tile
                    for ii in range(i, i_end):
                        for jj in range(j, j_end):
                            acc = 0.0
                            for kk in range(k, k_end):
                                acc += A[ii, kk] * B[kk, jj]
                            C[ii, jj] += acc

        return C

    @njit(fastmath=True, cache=True)
    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Root Mean Square Layer Normalization.

        RMSNorm normalizes by the RMS of the input without centering.
        Used in LLaMA and other modern architectures.

        Args:
            x: Input tensor (1D)
            weight: Learnable scale parameters
            eps: Small constant for numerical stability

        Returns:
            Normalized and scaled tensor
        """
        # Compute RMS
        rms = 0.0
        for i in range(len(x)):
            rms += x[i] * x[i]
        rms = np.sqrt(rms / len(x) + eps)

        # Normalize and scale
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = (x[i] / rms) * weight[i]

        return out

    @njit(fastmath=True, cache=True)
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.

        Uses the max-subtraction trick for numerical stability.

        Args:
            x: Input logits (1D)

        Returns:
            Probability distribution
        """
        # Find max for stability
        max_val = x[0]
        for i in range(1, len(x)):
            if x[i] > max_val:
                max_val = x[i]

        # Compute exp and sum
        out = np.empty_like(x)
        sum_exp = 0.0
        for i in range(len(x)):
            out[i] = np.exp(x[i] - max_val)
            sum_exp += out[i]

        # Normalize
        for i in range(len(x)):
            out[i] /= sum_exp

        return out

    @njit(fastmath=True, cache=True)
    def rope_embed(x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        """
        Rotary Position Embeddings (RoPE).

        Applies rotary position encoding to input embeddings.
        Used in LLaMA and other modern transformer architectures.

        Args:
            x: Input embedding (1D, even length)
            pos: Position index
            theta: Base frequency (default 10000.0)

        Returns:
            Position-encoded embedding
        """
        dim = len(x)
        out = np.empty_like(x)

        for i in range(0, dim, 2):
            freq = 1.0 / (theta ** (i / dim))
            cos_val = np.cos(pos * freq)
            sin_val = np.sin(pos * freq)

            out[i] = x[i] * cos_val - x[i + 1] * sin_val
            out[i + 1] = x[i] * sin_val + x[i + 1] * cos_val

        return out

    @njit(fastmath=True, cache=True)
    def silu(x: np.ndarray) -> np.ndarray:
        """
        SiLU (Swish) activation function.

        SiLU(x) = x * sigmoid(x)
        Used in LLaMA's SwiGLU FFN.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        out = np.empty_like(x)
        for i in range(len(x)):
            sig = 1.0 / (1.0 + np.exp(-x[i]))
            out[i] = x[i] * sig
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def gelu(x: np.ndarray) -> np.ndarray:
        """
        Gaussian Error Linear Unit activation.

        GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
        Approximated using tanh.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        out = np.empty_like(x)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

        for i in prange(len(x)):
            # Approximate GELU using tanh
            inner = sqrt_2_over_pi * (x[i] + 0.044715 * x[i] ** 3)
            out[i] = 0.5 * x[i] * (1.0 + np.tanh(inner))

        return out

    @njit(fastmath=True, cache=True)
    def attention_scores(q: np.ndarray, k_cache: np.ndarray,
                         head_dim: int) -> np.ndarray:
        """
        Compute attention scores between query and cached keys.

        Args:
            q: Query vector [head_dim]
            k_cache: Key cache [seq_len, head_dim]
            head_dim: Head dimension for scaling

        Returns:
            Attention scores [seq_len]
        """
        seq_len = k_cache.shape[0]
        scores = np.empty(seq_len, dtype=np.float32)
        scale = 1.0 / np.sqrt(head_dim)

        for i in range(seq_len):
            dot = 0.0
            for j in range(head_dim):
                dot += q[j] * k_cache[i, j]
            scores[i] = dot * scale

        return scores

    @njit(fastmath=True, cache=True)
    def weighted_sum(probs: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
        """
        Compute weighted sum of values.

        Args:
            probs: Attention probabilities [seq_len]
            v_cache: Value cache [seq_len, head_dim]

        Returns:
            Weighted sum [head_dim]
        """
        head_dim = v_cache.shape[1]
        out = np.zeros(head_dim, dtype=np.float32)

        for i in range(len(probs)):
            for j in range(head_dim):
                out[j] += probs[i] * v_cache[i, j]

        return out

else:
    # Fallback implementations without Numba SIMD

    def matmul_f32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication using NumPy."""
        return (A @ B).astype(np.float32)

    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMS normalization using NumPy."""
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return ((x / rms) * weight).astype(np.float32)

    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax using NumPy."""
        exp_x = np.exp(x - np.max(x))
        return (exp_x / exp_x.sum()).astype(np.float32)

    def rope_embed(x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        """Rotary position embeddings using NumPy."""
        dim = len(x)
        out = np.empty_like(x)

        for i in range(0, dim, 2):
            freq = 1.0 / (theta ** (i / dim))
            cos_val = np.cos(pos * freq)
            sin_val = np.sin(pos * freq)
            out[i] = x[i] * cos_val - x[i + 1] * sin_val
            out[i + 1] = x[i] * sin_val + x[i + 1] * cos_val

        return out.astype(np.float32)

    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) activation using NumPy."""
        return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)

    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation using NumPy."""
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
        return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)

    def attention_scores(q: np.ndarray, k_cache: np.ndarray,
                         head_dim: int) -> np.ndarray:
        """Compute attention scores using NumPy."""
        scale = 1.0 / np.sqrt(head_dim)
        return (k_cache @ q * scale).astype(np.float32)

    def weighted_sum(probs: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
        """Compute weighted sum using NumPy."""
        return (probs @ v_cache).astype(np.float32)


# Additional utility operators

def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """
    Standard Layer Normalization.

    Args:
        x: Input tensor
        weight: Scale parameters
        bias: Shift parameters
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    mean = np.mean(x)
    var = np.var(x)
    normalized = (x - mean) / np.sqrt(var + eps)
    return (normalized * weight + bias).astype(np.float32)


def apply_rope_batch(x: np.ndarray, positions: np.ndarray,
                     theta: float = 10000.0) -> np.ndarray:
    """
    Apply RoPE to a batch of embeddings.

    Args:
        x: Input embeddings [batch, dim]
        positions: Position indices [batch]
        theta: Base frequency

    Returns:
        Position-encoded embeddings [batch, dim]
    """
    batch_size, dim = x.shape
    out = np.empty_like(x)

    for b in range(batch_size):
        out[b] = rope_embed(x[b], int(positions[b]), theta)

    return out


def compute_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        Lower triangular mask [seq_len, seq_len]
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[mask == 1] = float('-inf')
    return mask.astype(np.float32)
