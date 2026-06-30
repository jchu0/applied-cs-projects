"""Optimizers for neural network training."""

import numpy as np
from typing import Iterator
from ..core.tensor import Tensor


class Optimizer:
    """Base optimizer class."""

    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        """Update parameters."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(
        self,
        parameters: Iterator[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity for momentum
        self._velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update parameters."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Momentum
            if self.momentum > 0:
                self._velocity[i] = self.momentum * self._velocity[i] + grad
                update = self._velocity[i]
            else:
                update = grad

            # Update parameters
            p.data -= self.lr * update


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        parameters: Iterator[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Moment estimates
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        """Update parameters."""
        self._t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Update moment estimates
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        parameters: Iterator[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        """Update parameters."""
        self._t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # Update moment estimates
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)

            # Update with decoupled weight decay
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        parameters: Iterator[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update parameters."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Update running average of squared gradients
            self._v[i] = self.alpha * self._v[i] + (1 - self.alpha) * (grad ** 2)

            # Update parameters
            p.data -= self.lr * grad / (np.sqrt(self._v[i]) + self.eps)
