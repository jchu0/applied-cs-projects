"""Neural network modules with automatic differentiation."""

import numpy as np
from typing import List, Iterator, Tuple, Optional
from abc import ABC, abstractmethod

from ..core.tensor import Tensor
from ..ops import relu, sigmoid, tanh, softmax, matmul, add


class Module(ABC):
    """Base class for neural network modules."""

    def __init__(self):
        self._parameters: List[Tensor] = []
        self._modules: List['Module'] = []
        self._training = True

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        pass

    def parameters(self) -> Iterator[Tensor]:
        """Iterate over all parameters."""
        for p in self._parameters:
            yield p
        for m in self._modules:
            yield from m.parameters()

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        for m in self._modules:
            m.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters():
            p.zero_grad()

    def _register_parameter(self, param: Tensor):
        """Register a parameter."""
        self._parameters.append(param)

    def _register_module(self, module: 'Module'):
        """Register a submodule."""
        self._modules.append(module)


class Linear(Module):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * std,
            requires_grad=True
        )
        self._register_parameter(self.weight)

        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
            self._register_parameter(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.weight)
        if self.bias is not None:
            out = add(out, self.bias)
        return out


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming initialization
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std,
            requires_grad=True
        )
        self._register_parameter(self.weight)

        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self._register_parameter(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        K = self.kernel_size

        # Pad input
        if self.padding > 0:
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            padded = x.data

        # Output dimensions
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        # Im2col for efficient convolution
        col = np.zeros((N, C, K, K, H_out, W_out))
        for i in range(K):
            i_max = i + self.stride * H_out
            for j in range(K):
                j_max = j + self.stride * W_out
                col[:, :, i, j, :, :] = padded[:, :, i:i_max:self.stride, j:j_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        weight_col = self.weight.data.reshape(self.out_channels, -1).T

        out = col @ weight_col
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1, 1)

        result = Tensor(out, requires_grad=x.requires_grad or self.weight.requires_grad)

        if result.requires_grad:
            def grad_fn(g):
                # Gradient w.r.t input
                # Simplified backward pass
                return np.zeros_like(x.data)
            result._set_grad_fn(grad_fn, [x])

        return result


class BatchNorm1d(Module):
    """Batch normalization for 1D inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self._register_parameter(self.gamma)
        self._register_parameter(self.beta)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if self._training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data

        result = Tensor(out, requires_grad=x.requires_grad)

        if x.requires_grad:
            def grad_fn(g):
                N = x.data.shape[0]
                std = np.sqrt(var + self.eps)
                x_hat = (x.data - mean) / std

                dgamma = (g * x_hat).sum(axis=0)
                dbeta = g.sum(axis=0)

                dx_hat = g * self.gamma.data
                dx = (1 / (N * std)) * (N * dx_hat - dx_hat.sum(axis=0) - x_hat * (dx_hat * x_hat).sum(axis=0))

                self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma
                self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta

                return dx
            result._set_grad_fn(grad_fn, [x])

        return result


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)
        self._register_parameter(self.gamma)
        self._register_parameter(self.beta)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data

        result = Tensor(out, requires_grad=x.requires_grad)

        if x.requires_grad:
            def grad_fn(g):
                return g * self.gamma.data / np.sqrt(var + self.eps)
            result._set_grad_fn(grad_fn, [x])

        return result


class Dropout(Module):
    """Dropout regularization."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x: Tensor) -> Tensor:
        if self._training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            out = x.data * self._mask
        else:
            out = x.data

        result = Tensor(out, requires_grad=x.requires_grad)

        if x.requires_grad and self._training:
            mask = self._mask

            def grad_fn(g):
                return g * mask
            result._set_grad_fn(grad_fn, [x])

        return result


# Activation modules
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.axis)


class Sequential(Module):
    """Sequential container for modules."""

    def __init__(self, *modules: Module):
        super().__init__()
        for m in modules:
            self._register_module(m)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules:
            x = module(x)
        return x


# Loss functions
class MSELoss(Module):
    """Mean squared error loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.data - target.data
        loss = np.mean(diff ** 2)

        result = Tensor(loss, requires_grad=pred.requires_grad)

        if pred.requires_grad:
            n = pred.size

            def grad_fn(g):
                return (2 * diff / n) * g
            result._set_grad_fn(grad_fn, [pred])

        return result


class CrossEntropyLoss(Module):
    """Cross entropy loss with softmax."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Softmax
        shifted = pred.data - np.max(pred.data, axis=-1, keepdims=True)
        exp_pred = np.exp(shifted)
        probs = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

        # Cross entropy
        n = pred.shape[0]

        if target.ndim == 1:
            # Class indices
            log_probs = -np.log(probs[np.arange(n), target.data.astype(int)] + 1e-8)
        else:
            # One-hot
            log_probs = -np.sum(target.data * np.log(probs + 1e-8), axis=-1)

        loss = np.mean(log_probs)

        result = Tensor(loss, requires_grad=pred.requires_grad)

        if pred.requires_grad:
            def grad_fn(g):
                grad = probs.copy()
                if target.ndim == 1:
                    grad[np.arange(n), target.data.astype(int)] -= 1
                else:
                    grad -= target.data
                return (grad / n) * g
            result._set_grad_fn(grad_fn, [pred])

        return result
