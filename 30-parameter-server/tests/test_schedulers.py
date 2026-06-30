"""Tests for learning rate schedulers."""

import pytest
import math

from paramserver.optimizer.schedulers import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    WarmupLR,
    CosineWarmupLR,
    PolynomialLR,
    OneCycleLR,
)


class TestStepLR:
    """Tests for StepLR scheduler."""

    def test_step_decay(self):
        """Test step-based decay."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

        # Steps 1-10: lr = 0.1 (step_count 1-10 // 10 = 0)
        for i in range(10):
            lr = scheduler.step()
            # After step, step_count = i+1, so lr = base_lr * gamma^((i+1)//10)
            expected = 0.1 * (0.1 ** ((i + 1) // 10))
            assert abs(lr - expected) < 1e-6

        # Steps 11-20: step_count 11-20, lr = base_lr * gamma^1 = 0.01
        for i in range(10):
            lr = scheduler.step()
            # step_count = 10 + i + 1 = 11 to 20
            step_count = 10 + i + 1
            expected = 0.1 * (0.1 ** (step_count // 10))
            assert abs(lr - expected) < 1e-6

        # Steps 21-30: step_count 21-30, lr = base_lr * gamma^2 = 0.001
        for i in range(10):
            lr = scheduler.step()
            step_count = 20 + i + 1
            expected = 0.1 * (0.1 ** (step_count // 10))
            assert abs(lr - expected) < 1e-6

    def test_get_lr(self):
        """Test get_lr without stepping."""
        scheduler = StepLR(base_lr=0.1, step_size=5, gamma=0.5)

        assert scheduler.get_lr() == 0.1
        scheduler._step_count = 5
        assert scheduler.get_lr() == 0.05

    def test_reset(self):
        """Test reset."""
        scheduler = StepLR(base_lr=0.1, step_size=5, gamma=0.5)
        for _ in range(10):
            scheduler.step()

        scheduler.reset()
        assert scheduler.step_count == 0
        assert scheduler.get_lr() == 0.1


class TestMultiStepLR:
    """Tests for MultiStepLR scheduler."""

    def test_milestone_decay(self):
        """Test decay at milestones."""
        scheduler = MultiStepLR(
            base_lr=0.1,
            milestones=[10, 20, 30],
            gamma=0.1,
        )

        # Before first milestone
        assert scheduler.get_lr() == 0.1

        # Step to milestone
        for _ in range(10):
            scheduler.step()
        assert abs(scheduler.get_lr() - 0.01) < 1e-8

        # Step to second milestone
        for _ in range(10):
            scheduler.step()
        assert abs(scheduler.get_lr() - 0.001) < 1e-8

    def test_unsorted_milestones(self):
        """Test that milestones are sorted internally."""
        scheduler = MultiStepLR(
            base_lr=0.1,
            milestones=[30, 10, 20],
            gamma=0.1,
        )
        assert scheduler.milestones == [10, 20, 30]


class TestExponentialLR:
    """Tests for ExponentialLR scheduler."""

    def test_exponential_decay(self):
        """Test exponential decay."""
        scheduler = ExponentialLR(base_lr=1.0, gamma=0.9)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_lr())

        # Each LR should be 0.9 * previous
        for i in range(1, len(lrs)):
            ratio = lrs[i] / lrs[i - 1]
            assert abs(ratio - 0.9) < 1e-6

    def test_decay_formula(self):
        """Test exact decay formula."""
        scheduler = ExponentialLR(base_lr=0.1, gamma=0.5)

        for step in range(5):
            scheduler.step()
            # After step, step_count = step+1
            expected = 0.1 * (0.5 ** (step + 1))
            assert abs(scheduler.get_lr() - expected) < 1e-8


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_cosine_curve(self):
        """Test cosine annealing curve."""
        scheduler = CosineAnnealingLR(
            base_lr=1.0,
            T_max=100,
            eta_min=0.0,
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_lr())

        # Should start near 1.0 and end near 0.0
        assert lrs[0] > 0.9
        assert lrs[-1] < 0.1

        # Should be monotonically decreasing
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-6

    def test_minimum_lr(self):
        """Test minimum learning rate."""
        scheduler = CosineAnnealingLR(
            base_lr=1.0,
            T_max=50,
            eta_min=0.1,
        )

        for _ in range(100):
            scheduler.step()

        assert scheduler.get_lr() >= 0.1


class TestWarmupLR:
    """Tests for WarmupLR scheduler."""

    def test_linear_warmup(self):
        """Test linear warmup phase."""
        scheduler = WarmupLR(base_lr=1.0, warmup_steps=10)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.last_lr)

        # After first step, step_count=1, lr = base_lr * (1+1)/10 = 0.2
        assert abs(lrs[0] - 0.2) < 1e-6
        # After 10th step, step_count=10, lr = base_lr * (10+1)/10 = 1.1, capped at base_lr
        # Actually at step 10, we're past warmup, so lr = base_lr = 1.0
        assert abs(lrs[-1] - 1.0) < 1e-6

    def test_warmup_then_constant(self):
        """Test constant LR after warmup."""
        scheduler = WarmupLR(base_lr=0.1, warmup_steps=5)

        for _ in range(20):
            scheduler.step()

        # Should be at base_lr after warmup
        assert scheduler.get_lr() == 0.1

    def test_warmup_with_wrapped(self):
        """Test warmup with wrapped scheduler."""
        inner = StepLR(base_lr=0.1, step_size=5, gamma=0.1)
        scheduler = WarmupLR(
            base_lr=0.1,
            warmup_steps=10,
            wrapped_scheduler=inner,
        )

        # Warmup phase
        for _ in range(10):
            scheduler.step()
        assert abs(scheduler.last_lr - 0.1) < 1e-6

        # After warmup, StepLR takes over
        for _ in range(5):
            scheduler.step()
        assert abs(scheduler.last_lr - 0.01) < 1e-6


class TestCosineWarmupLR:
    """Tests for CosineWarmupLR scheduler."""

    def test_warmup_then_cosine(self):
        """Test warmup followed by cosine decay."""
        scheduler = CosineWarmupLR(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=110,
            eta_min=0.0,
        )

        lrs = []
        for _ in range(110):
            scheduler.step()
            lrs.append(scheduler.last_lr)

        # Warmup: increases to 1.0
        assert lrs[9] == 1.0

        # Cosine: decreases to 0.0
        assert lrs[-1] < 0.1

    def test_minimum_lr(self):
        """Test minimum learning rate."""
        scheduler = CosineWarmupLR(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=50,
            eta_min=0.1,
        )

        for _ in range(100):
            scheduler.step()

        assert scheduler.get_lr() >= 0.1


class TestPolynomialLR:
    """Tests for PolynomialLR scheduler."""

    def test_linear_decay(self):
        """Test linear decay (power=1)."""
        scheduler = PolynomialLR(
            base_lr=1.0,
            total_steps=100,
            power=1.0,
            eta_min=0.0,
        )

        for step in range(100):
            scheduler.step()
            # After step, step_count = step+1
            expected = 1.0 * (1 - (step + 1) / 100)
            assert abs(scheduler.get_lr() - expected) < 1e-6

    def test_quadratic_decay(self):
        """Test quadratic decay (power=2)."""
        scheduler = PolynomialLR(
            base_lr=1.0,
            total_steps=100,
            power=2.0,
            eta_min=0.0,
        )

        for step in range(50):
            scheduler.step()
            # After step, step_count = step+1
            expected = ((1 - (step + 1) / 100) ** 2)
            assert abs(scheduler.get_lr() - expected) < 1e-6

    def test_minimum_lr(self):
        """Test minimum learning rate."""
        scheduler = PolynomialLR(
            base_lr=1.0,
            total_steps=50,
            power=1.0,
            eta_min=0.1,
        )

        for _ in range(100):
            scheduler.step()

        assert scheduler.get_lr() >= 0.1


class TestOneCycleLR:
    """Tests for OneCycleLR scheduler."""

    def test_one_cycle_shape(self):
        """Test one-cycle policy shape."""
        scheduler = OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.3,
            div_factor=25.0,
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.last_lr)

        # Should start low
        assert lrs[0] < 0.1

        # Should peak around 30%
        peak_idx = lrs.index(max(lrs))
        assert 25 <= peak_idx <= 35

        # Should end low
        assert lrs[-1] < 0.01

    def test_initial_lr(self):
        """Test initial learning rate calculation."""
        scheduler = OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            div_factor=10.0,
        )
        assert scheduler.base_lr == 0.1  # max_lr / div_factor

    def test_final_lr(self):
        """Test final learning rate calculation."""
        scheduler = OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        initial_lr = 1.0 / 25.0
        final_lr = initial_lr / 10000.0

        for _ in range(100):
            scheduler.step()

        assert abs(scheduler.get_lr() - final_lr) < 1e-8


class TestSchedulerProperties:
    """Tests for common scheduler properties."""

    def test_step_count(self):
        """Test step count tracking."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

        assert scheduler.step_count == 0
        scheduler.step()
        assert scheduler.step_count == 1
        scheduler.step()
        assert scheduler.step_count == 2

    def test_last_lr(self):
        """Test last_lr property."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

        assert scheduler.last_lr == 0.1
        scheduler.step()
        assert scheduler.last_lr == 0.1

    def test_reset(self):
        """Test reset functionality."""
        scheduler = CosineWarmupLR(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=100,
        )

        for _ in range(50):
            scheduler.step()

        scheduler.reset()
        assert scheduler.step_count == 0
        assert scheduler.last_lr == scheduler.base_lr
