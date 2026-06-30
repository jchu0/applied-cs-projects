"""Pipeline parallelism implementation."""

import numpy as np
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue

from ..core.context import ProcessGroup

logger = logging.getLogger(__name__)


@dataclass
class MicroBatch:
    """A micro-batch for pipeline processing."""
    index: int
    data: Any
    target: Any = None
    loss: float = 0.0
    activations: List[np.ndarray] = field(default_factory=list)
    gradients: List[np.ndarray] = field(default_factory=list)


class PipelineSchedule(Enum):
    """Pipeline scheduling strategies."""
    GPIPE = auto()       # Fill-drain schedule
    INTERLEAVED = auto() # 1F1B interleaved
    ASYNC = auto()       # Asynchronous


@dataclass
class PipelineStage:
    """A stage in the pipeline."""
    stage_id: int
    module: Any
    device: int = 0
    input_queue: Queue = field(default_factory=Queue)
    output_queue: Queue = field(default_factory=Queue)
    grad_queue: Queue = field(default_factory=Queue)

    def forward(self, micro_batch: MicroBatch) -> MicroBatch:
        """Forward pass for this stage."""
        if hasattr(self.module, '__call__'):
            output = self.module(micro_batch.data)
        else:
            output = micro_batch.data

        # Save activation for backward
        micro_batch.activations.append(micro_batch.data)
        micro_batch.data = output
        return micro_batch

    def backward(self, micro_batch: MicroBatch, grad: np.ndarray) -> np.ndarray:
        """Backward pass for this stage."""
        # Get saved activation
        if micro_batch.activations:
            activation = micro_batch.activations.pop()
        else:
            activation = None

        # Compute gradient (simplified)
        input_grad = grad

        # Store gradient
        micro_batch.gradients.insert(0, grad)

        return input_grad


class PipelineParallel:
    """
    Pipeline parallel training.

    Splits model across stages and processes micro-batches.

    Features:
    - Multiple scheduling strategies
    - Activation checkpointing
    - Memory optimization
    """

    def __init__(
        self,
        modules: List[Any],
        num_microbatches: int = 1,
        schedule: PipelineSchedule = PipelineSchedule.GPIPE,
        process_group: ProcessGroup = None,
        checkpoint_activations: bool = False
    ):
        self.num_microbatches = num_microbatches
        self.schedule = schedule
        self.process_group = process_group
        self.checkpoint_activations = checkpoint_activations

        # Create stages
        self.stages = []
        for i, module in enumerate(modules):
            stage = PipelineStage(
                stage_id=i,
                module=module,
                device=i % 8
            )
            self.stages.append(stage)

        self.num_stages = len(self.stages)

    def forward(self, batch_data: Any, batch_target: Any = None) -> List[MicroBatch]:
        """Forward pass through pipeline."""
        # Split into micro-batches
        if isinstance(batch_data, np.ndarray):
            micro_batch_data = np.array_split(batch_data, self.num_microbatches)
        else:
            micro_batch_data = [batch_data] * self.num_microbatches

        if batch_target is not None and isinstance(batch_target, np.ndarray):
            micro_batch_target = np.array_split(batch_target, self.num_microbatches)
        else:
            micro_batch_target = [batch_target] * self.num_microbatches

        # Create micro-batches
        micro_batches = [
            MicroBatch(i, data, target)
            for i, (data, target) in enumerate(zip(micro_batch_data, micro_batch_target))
        ]

        # Run schedule
        if self.schedule == PipelineSchedule.GPIPE:
            return self._gpipe_schedule(micro_batches)
        elif self.schedule == PipelineSchedule.INTERLEAVED:
            return self._interleaved_schedule(micro_batches)
        else:
            return self._gpipe_schedule(micro_batches)

    def _gpipe_schedule(self, micro_batches: List[MicroBatch]) -> List[MicroBatch]:
        """GPipe fill-drain schedule."""
        # Forward pass - fill pipeline
        for mb in micro_batches:
            for stage in self.stages:
                mb = stage.forward(mb)

        return micro_batches

    def _interleaved_schedule(self, micro_batches: List[MicroBatch]) -> List[MicroBatch]:
        """1F1B interleaved schedule for better memory."""
        num_warmup = min(self.num_stages - 1, len(micro_batches))
        results = []

        # Warmup phase
        in_flight = []
        for i in range(num_warmup):
            mb = micro_batches[i]
            for stage in self.stages:
                mb = stage.forward(mb)
            in_flight.append(mb)

        # Steady state - 1F1B
        for i in range(num_warmup, len(micro_batches)):
            # Forward for new micro-batch
            mb = micro_batches[i]
            for stage in self.stages:
                mb = stage.forward(mb)

            # Add to in-flight
            in_flight.append(mb)

            # Could do backward here for oldest in-flight
            if in_flight:
                results.append(in_flight.pop(0))

        # Drain remaining
        results.extend(in_flight)

        return results

    def backward(self, micro_batches: List[MicroBatch], loss_fn: Callable = None):
        """Backward pass through pipeline."""
        for mb in reversed(micro_batches):
            # Compute loss if provided
            if loss_fn and mb.target is not None:
                mb.loss = loss_fn(mb.data, mb.target)
                grad = np.ones_like(mb.data)  # d(loss)/d(output)
            else:
                grad = np.ones_like(mb.data)

            # Backward through stages
            for stage in reversed(self.stages):
                grad = stage.backward(mb, grad)

    def get_loss(self, micro_batches: List[MicroBatch]) -> float:
        """Get average loss across micro-batches."""
        if not micro_batches:
            return 0.0
        return np.mean([mb.loss for mb in micro_batches])


class GPipeSchedule:
    """GPipe-style fill-drain schedule."""

    def __init__(self, num_stages: int, num_microbatches: int):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

    def get_schedule(self) -> List[Tuple[int, int, str]]:
        """Get schedule as list of (stage, microbatch, 'F'/'B')."""
        schedule = []

        # Forward passes
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                schedule.append((stage, mb, 'F'))

        # Backward passes
        for mb in reversed(range(self.num_microbatches)):
            for stage in reversed(range(self.num_stages)):
                schedule.append((stage, mb, 'B'))

        return schedule


class InterleavedSchedule:
    """1F1B interleaved schedule."""

    def __init__(self, num_stages: int, num_microbatches: int):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

    def get_schedule(self) -> List[Tuple[int, int, str]]:
        """Get interleaved schedule."""
        schedule = []
        num_warmup = min(self.num_stages - 1, self.num_microbatches)

        # Warmup
        for mb in range(num_warmup):
            for stage in range(self.num_stages):
                schedule.append((stage, mb, 'F'))

        # Steady state
        for mb in range(num_warmup, self.num_microbatches):
            for stage in range(self.num_stages):
                schedule.append((stage, mb, 'F'))
            for stage in reversed(range(self.num_stages)):
                schedule.append((stage, mb - num_warmup, 'B'))

        # Cooldown
        for mb in range(self.num_microbatches - num_warmup, self.num_microbatches):
            for stage in reversed(range(self.num_stages)):
                schedule.append((stage, mb, 'B'))

        return schedule
