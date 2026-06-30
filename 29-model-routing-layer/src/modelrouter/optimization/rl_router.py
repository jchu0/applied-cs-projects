"""Reinforcement learning-based routing for model routing layer."""

import logging
import numpy as np
from collections import deque
from typing import Optional

from ..schemas import InferenceRequest, WorkerInfo, RLExperience

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Replay buffer for RL experiences with O(1) lookup by request_id."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: deque[RLExperience] = deque(maxlen=capacity)
        self._index: dict[str, RLExperience] = {}

    def add(self, experience: RLExperience):
        """Add experience to buffer."""
        if len(self._buffer) == self.capacity:
            # Remove oldest from index
            oldest = self._buffer[0]
            self._index.pop(oldest.request_id, None)
        self._buffer.append(experience)
        self._index[experience.request_id] = experience

    def get(self, request_id: str) -> Optional[RLExperience]:
        """Get experience by request_id."""
        return self._index.get(request_id)

    def update_reward(self, request_id: str, reward: float) -> bool:
        """Update reward for an experience. Returns True if found."""
        exp = self._index.get(request_id)
        if exp is None:
            return False
        exp.reward = reward
        exp.done = True
        return True

    def sample(self, batch_size: int) -> list[RLExperience]:
        """Sample a random mini-batch from the buffer."""
        if len(self._buffer) == 0:
            return []
        batch_size = min(batch_size, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class DQNPolicy:
    """2-layer feedforward DQN with numpy-only implementation.

    Architecture: state_dim -> hidden_dim (ReLU) -> hidden_dim (ReLU) -> action_dim
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Xavier initialization
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / (state_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / (hidden_dim + action_dim))
        self.b3 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass returning Q-values for all actions.

        Args:
            state: State vector of shape (state_dim,) or (batch, state_dim)

        Returns:
            Q-values of shape (action_dim,) or (batch, action_dim)
        """
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        q_values = h2 @ self.W3 + self.b3
        return q_values

    def select_action(self, state: np.ndarray, num_valid_actions: int = None) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: State vector
            num_valid_actions: Number of valid actions (limits action space)

        Returns:
            Selected action index
        """
        if num_valid_actions is None:
            num_valid_actions = self.action_dim

        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, num_valid_actions))

        q_values = self.forward(state)
        # Only consider valid actions
        valid_q = q_values[:num_valid_actions]
        return int(np.argmax(valid_q))

    def update(self, experiences: list) -> float:
        """Mini-batch gradient descent update.

        Args:
            experiences: List of RLExperience objects with done=True

        Returns:
            Mean loss
        """
        if not experiences:
            return 0.0

        # Build batch arrays
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences], dtype=int)
        rewards = np.array([e.reward for e in experiences])

        # Forward pass
        h1 = np.maximum(0, states @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        q_values = h2 @ self.W3 + self.b3

        # Compute targets (no next-state for terminal experiences)
        targets = q_values.copy()
        for i, action in enumerate(actions):
            targets[i, action] = rewards[i]

        # Loss: MSE between q_values and targets (only on taken actions)
        batch_size = len(experiences)
        loss_per_sample = np.zeros(batch_size)
        for i in range(batch_size):
            loss_per_sample[i] = (q_values[i, actions[i]] - targets[i, actions[i]]) ** 2
        loss = np.mean(loss_per_sample)

        # Backpropagation
        # dL/dq = 2 * (q - target) / batch_size, but only for taken actions
        dq = np.zeros_like(q_values)
        for i in range(batch_size):
            dq[i, actions[i]] = 2.0 * (q_values[i, actions[i]] - rewards[i]) / batch_size

        # Layer 3
        dW3 = h2.T @ dq
        db3 = np.sum(dq, axis=0)
        dh2 = dq @ self.W3.T

        # ReLU derivative for layer 2
        dh2 = dh2 * (h2 > 0)

        # Layer 2
        dW2 = h1.T @ dh2
        db2 = np.sum(dh2, axis=0)
        dh1 = dh2 @ self.W2.T

        # ReLU derivative for layer 1
        dh1 = dh1 * (h1 > 0)

        # Layer 1
        dW1 = states.T @ dh1
        db1 = np.sum(dh1, axis=0)

        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return float(loss)


class RLRouter:
    """RL-based routing that learns optimal worker selection.

    State vector: 4 request features + max_workers*6 worker features.
    Action space: discrete [0, max_workers).
    """

    def __init__(
        self,
        policy: DQNPolicy,
        experience_buffer: ExperienceBuffer,
        max_workers: int = 16,
    ):
        self.policy = policy
        self.experience_buffer = experience_buffer
        self.max_workers = max_workers

    def route(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo],
    ) -> WorkerInfo:
        """Route request using RL policy.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        state = self._extract_state(request, workers)
        action = self.policy.select_action(state, num_valid_actions=len(workers))

        # Store experience (reward will be updated later)
        experience = RLExperience(
            request_id=request.request_id,
            state=state,
            action=action,
        )
        self.experience_buffer.add(experience)

        return workers[action]

    def update(self, request_id: str, reward: float) -> bool:
        """Feed reward back for a completed request.

        Args:
            request_id: The request that completed
            reward: Computed reward signal

        Returns:
            True if experience was found and updated
        """
        return self.experience_buffer.update_reward(request_id, reward)

    @staticmethod
    def compute_reward(
        latency_ms: float,
        sla_deadline_ms: float = None,
    ) -> float:
        """Compute reward signal for a completed request.

        Reward = -latency/1000 + SLA_bonus
        SLA_bonus = +1.0 if met deadline, -2.0 if missed

        Args:
            latency_ms: Observed latency
            sla_deadline_ms: SLA deadline (optional)

        Returns:
            Reward value
        """
        reward = -latency_ms / 1000.0
        if sla_deadline_ms is not None:
            if latency_ms <= sla_deadline_ms:
                reward += 1.0
            else:
                reward -= 2.0
        return reward

    def _extract_state(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo],
    ) -> np.ndarray:
        """Extract state vector from request and workers.

        State layout:
          [0:4] - request features: priority, estimated_tokens/1000,
                  temperature, sla_deadline_ms/10000
          [4:4+max_workers*6] - per-worker features (zero-padded):
                  current_load, queue_depth/100, tokens_in_flight/token_budget,
                  gpu_util/100, gpu_mem_used/gpu_mem_total, performance_factor

        Total: 4 + max_workers*6 = 100 dims (for max_workers=16)
        """
        state = np.zeros(4 + self.max_workers * 6, dtype=np.float32)

        # Request features
        state[0] = request.priority.value / 4.0  # Normalize to [0, 1]
        state[1] = request.estimated_tokens / 1000.0
        state[2] = request.temperature
        state[3] = (request.sla_deadline_ms or 0) / 10000.0

        # Worker features
        for i, w in enumerate(workers):
            if i >= self.max_workers:
                break
            offset = 4 + i * 6
            state[offset] = w.current_load
            state[offset + 1] = w.queue_depth / 100.0
            token_budget = max(w.token_budget, 1)
            state[offset + 2] = w.tokens_in_flight / token_budget
            state[offset + 3] = w.gpu_info.utilization_percent / 100.0
            mem_total = max(w.gpu_info.memory_total_mb, 1)
            state[offset + 4] = w.gpu_info.memory_used_mb / mem_total
            state[offset + 5] = w.performance_factor

        return state

    def train(self, batch_size: int = 32) -> float:
        """Train the policy on a batch from the replay buffer.

        Only trains on completed experiences (done=True).

        Args:
            batch_size: Mini-batch size

        Returns:
            Training loss
        """
        samples = self.experience_buffer.sample(batch_size)
        completed = [s for s in samples if s.done]
        if not completed:
            return 0.0
        return self.policy.update(completed)
