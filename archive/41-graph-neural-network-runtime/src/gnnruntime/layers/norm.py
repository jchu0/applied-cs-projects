"""Normalization layers for GNN runtime."""

import numpy as np
from typing import Optional


class BatchNorm:
    """Batch normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Normalized tensor
        """
        if self.training:
            # Compute batch statistics
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


class LayerNorm:
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Learnable parameters (only used if elementwise_affine=True)
        if elementwise_affine:
            self.gamma = np.ones(normalized_shape)
            self.beta = np.zeros(normalized_shape)
        else:
            self.gamma = None
            self.beta = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (..., normalized_shape)

        Returns:
            Normalized tensor
        """
        # Normalize over last dimension using sample std (ddof=0)
        mean = np.mean(x, axis=-1, keepdims=True)
        # Use std for proper normalization
        std = np.std(x, axis=-1, keepdims=True)

        # Only add eps when std is very close to zero
        x_norm = (x - mean) / np.where(std > self.eps, std, std + self.eps)

        if self.elementwise_affine and self.gamma is not None:
            return self.gamma * x_norm + self.beta
        return x_norm


class GraphNorm:
    """
    Graph-level normalization.

    Normalizes node features within each graph in a batch.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5
    ):
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def __call__(
        self,
        x: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, num_features)
            batch: Batch assignment for each node

        Returns:
            Normalized features
        """
        if batch is None:
            # Single graph - normalize over all nodes
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
        else:
            # Multiple graphs - normalize per graph
            x_norm = np.zeros_like(x)
            unique_batch = np.unique(batch)

            for b in unique_batch:
                mask = batch == b
                x_b = x[mask]
                mean = np.mean(x_b, axis=0, keepdims=True)
                var = np.var(x_b, axis=0, keepdims=True)
                x_norm[mask] = (x_b - mean) / np.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta


class InstanceNorm:
    """Instance normalization for graphs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5
    ):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def __call__(
        self,
        x: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass - same as GraphNorm."""
        if batch is None:
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            x_norm = (x - mean) / (std + self.eps)
        else:
            x_norm = np.zeros_like(x)
            unique_batch = np.unique(batch)

            for b in unique_batch:
                mask = batch == b
                x_b = x[mask]
                mean = np.mean(x_b, axis=0, keepdims=True)
                std = np.std(x_b, axis=0, keepdims=True)
                x_norm[mask] = (x_b - mean) / (std + self.eps)

        return self.gamma * x_norm + self.beta
