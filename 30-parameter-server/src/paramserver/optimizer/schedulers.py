"""Learning rate schedulers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import math


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers.

    Schedulers adjust the learning rate during training based on
    step count, epoch, or other criteria.
    """

    def __init__(self, base_lr: float):
        """Initialize scheduler.

        Args:
            base_lr: Initial/base learning rate.
        """
        self.base_lr = base_lr
        self._step_count = 0
        self._last_lr = base_lr

    @abstractmethod
    def get_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate.
        """
        pass

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns:
            New learning rate.
        """
        self._step_count += 1
        self._last_lr = self.get_lr()
        return self._last_lr

    @property
    def last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self._last_lr

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self._step_count = 0
        self._last_lr = self.base_lr


class StepLR(LRScheduler):
    """Step learning rate scheduler.

    Decays the learning rate by gamma every step_size steps.

    lr = base_lr * gamma^(step // step_size)
    """

    def __init__(
        self,
        base_lr: float,
        step_size: int,
        gamma: float = 0.1,
    ):
        """Initialize StepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay.
        """
        super().__init__(base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.base_lr * (self.gamma ** (self._step_count // self.step_size))


class MultiStepLR(LRScheduler):
    """Multi-step learning rate scheduler.

    Decays learning rate by gamma at specified milestones.
    """

    def __init__(
        self,
        base_lr: float,
        milestones: List[int],
        gamma: float = 0.1,
    ):
        """Initialize MultiStepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            milestones: List of step indices at which to decay.
            gamma: Multiplicative factor of learning rate decay.
        """
        super().__init__(base_lr)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self) -> float:
        """Get current learning rate."""
        num_decays = sum(1 for m in self.milestones if self._step_count >= m)
        return self.base_lr * (self.gamma ** num_decays)


class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler.

    Decays learning rate by gamma every step.

    lr = base_lr * gamma^step
    """

    def __init__(
        self,
        base_lr: float,
        gamma: float = 0.99,
    ):
        """Initialize ExponentialLR scheduler.

        Args:
            base_lr: Initial learning rate.
            gamma: Multiplicative factor of learning rate decay.
        """
        super().__init__(base_lr)
        self.gamma = gamma

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.base_lr * (self.gamma ** self._step_count)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    Decays learning rate following a cosine curve from base_lr to eta_min.

    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step / T_max))
    """

    def __init__(
        self,
        base_lr: float,
        T_max: int,
        eta_min: float = 0.0,
    ):
        """Initialize CosineAnnealingLR scheduler.

        Args:
            base_lr: Initial learning rate.
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
        """
        super().__init__(base_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._step_count >= self.T_max:
            return self.eta_min

        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._step_count / self.T_max)
        )


class WarmupLR(LRScheduler):
    """Linear warmup learning rate scheduler.

    Linearly increases learning rate from 0 to base_lr over warmup_steps,
    then delegates to an optional wrapped scheduler.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        wrapped_scheduler: Optional[LRScheduler] = None,
    ):
        """Initialize WarmupLR scheduler.

        Args:
            base_lr: Target learning rate after warmup.
            warmup_steps: Number of warmup steps.
            wrapped_scheduler: Optional scheduler to use after warmup.
        """
        super().__init__(base_lr)
        self.warmup_steps = warmup_steps
        self.wrapped_scheduler = wrapped_scheduler

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self._step_count + 1) / self.warmup_steps

        if self.wrapped_scheduler is not None:
            # Delegate to wrapped scheduler
            # Adjust step count for the wrapped scheduler
            wrapped_step = self._step_count - self.warmup_steps
            # Temporarily set step count
            old_step = self.wrapped_scheduler._step_count
            self.wrapped_scheduler._step_count = wrapped_step
            lr = self.wrapped_scheduler.get_lr()
            self.wrapped_scheduler._step_count = old_step
            return lr

        return self.base_lr

    def reset(self) -> None:
        """Reset scheduler and wrapped scheduler."""
        super().reset()
        if self.wrapped_scheduler is not None:
            self.wrapped_scheduler.reset()


class CosineWarmupLR(LRScheduler):
    """Cosine decay with linear warmup.

    Combines linear warmup with cosine annealing decay.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
    ):
        """Initialize CosineWarmupLR scheduler.

        Args:
            base_lr: Peak learning rate (after warmup).
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
            eta_min: Minimum learning rate.
        """
        super().__init__(base_lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self._step_count + 1) / self.warmup_steps

        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        current_decay_step = self._step_count - self.warmup_steps

        if current_decay_step >= decay_steps:
            return self.eta_min

        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * current_decay_step / decay_steps)
        )


class PolynomialLR(LRScheduler):
    """Polynomial decay learning rate scheduler.

    Decays learning rate using polynomial function.

    lr = (base_lr - eta_min) * (1 - step/total_steps)^power + eta_min
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        power: float = 1.0,
        eta_min: float = 0.0,
    ):
        """Initialize PolynomialLR scheduler.

        Args:
            base_lr: Initial learning rate.
            total_steps: Total number of steps.
            power: Power of polynomial decay.
            eta_min: Minimum learning rate.
        """
        super().__init__(base_lr)
        self.total_steps = total_steps
        self.power = power
        self.eta_min = eta_min

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._step_count >= self.total_steps:
            return self.eta_min

        decay_factor = (1 - self._step_count / self.total_steps) ** self.power
        return (self.base_lr - self.eta_min) * decay_factor + self.eta_min


class OneCycleLR(LRScheduler):
    """One-cycle learning rate policy.

    Implements the 1cycle policy: linear warmup to max_lr, then
    cosine annealing down to min_lr.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ):
        """Initialize OneCycleLR scheduler.

        Args:
            max_lr: Maximum learning rate.
            total_steps: Total number of training steps.
            pct_start: Percentage of cycle spent increasing LR.
            div_factor: Determines initial LR (max_lr / div_factor).
            final_div_factor: Determines final LR (initial_lr / final_div_factor).
        """
        initial_lr = max_lr / div_factor
        super().__init__(initial_lr)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        self.min_lr = initial_lr / final_div_factor

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._step_count < self.step_up:
            # Linear warmup phase
            return self.base_lr + (self.max_lr - self.base_lr) * (
                self._step_count / self.step_up
            )
        else:
            # Cosine annealing phase
            down_step = self._step_count - self.step_up
            if down_step >= self.step_down:
                return self.min_lr

            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * down_step / self.step_down)
            )
