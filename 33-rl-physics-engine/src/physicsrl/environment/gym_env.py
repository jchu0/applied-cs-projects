"""Gym-compatible RL environment wrapper."""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional

from ..core.bodies import Model, State
from ..integration.integrator import Integrator


class PhysicsEnvironment:
    """
    Gym-compatible environment wrapper for physics simulation.
    """

    def __init__(self, model: Model):
        self.model = model
        self.integrator = Integrator(model)
        self.state = State.create(model)

        # Observation and action spaces
        self.observation_dim = self._compute_obs_dim()
        self.action_dim = model.nu

        # Episode settings
        self.max_episode_steps = 1000
        self._step_count = 0

        # Domain randomization settings
        self.randomize_mass = True
        self.randomize_friction = True
        self.mass_range = (0.9, 1.1)
        self.friction_range = (0.8, 1.2)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.state = State.create(self.model)
        self._step_count = 0

        # Apply domain randomization
        if self.randomize_mass or self.randomize_friction:
            self._apply_domain_randomization()

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step in environment.

        Args:
            action: Control inputs (nu,)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Step simulation
        self.state = self.integrator.step(self.state, action)
        self._step_count += 1

        # Compute observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps

        info = {
            'time': self.state.time,
            'n_contacts': len(self.state.contacts),
            'step_count': self._step_count
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation from state."""
        obs_parts = []

        # Joint positions
        obs_parts.append(self.state.qpos)

        # Joint velocities
        obs_parts.append(self.state.qvel)

        # Body positions (optional, can be enabled)
        # obs_parts.append(self.state.xpos.flatten())

        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state. Override in subclass."""
        # Default: penalize control effort
        ctrl_cost = 0.01 * np.sum(action ** 2)
        return -ctrl_cost

    def _check_termination(self) -> bool:
        """Check if episode should terminate. Override in subclass."""
        return False

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension."""
        return self.model.nq + self.model.nv

    def _apply_domain_randomization(self):
        """Apply domain randomization for sim2real."""
        if self.randomize_mass:
            for body in self.model.bodies:
                factor = np.random.uniform(*self.mass_range)
                body.inertia.mass *= factor
                body.inertia.inertia *= factor

        if self.randomize_friction:
            for body in self.model.bodies:
                for geom in body.geoms:
                    geom.friction *= np.random.uniform(*self.friction_range)

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        """Get observation space shape."""
        return (self.observation_dim,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        """Get action space shape."""
        return (self.action_dim,)


class BatchedEnvironment:
    """
    Batched environment for parallel simulation.
    """

    def __init__(self, model: Model, num_envs: int):
        self.num_envs = num_envs
        self.model = model

        # Create independent environment copies
        self.envs = [PhysicsEnvironment(model) for _ in range(num_envs)]

        # Observation and action dimensions
        self.observation_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset all environments."""
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs

        obs = np.stack([
            env.reset(seed=seeds[i]) for i, env in enumerate(self.envs)
        ])
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        results = [
            env.step(actions[i]) for i, env in enumerate(self.envs)
        ]

        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminated = np.array([r[2] for r in results], dtype=bool)
        truncated = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]

        # Auto-reset terminated environments
        for i in range(self.num_envs):
            if terminated[i] or truncated[i]:
                obs[i] = self.envs[i].reset()

        return obs, rewards, terminated, truncated, infos

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        """Get observation space shape per environment."""
        return (self.observation_dim,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        """Get action space shape per environment."""
        return (self.action_dim,)


# Example environment subclasses
class InvertedPendulumEnv(PhysicsEnvironment):
    """Inverted pendulum balancing task."""

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward for keeping pendulum upright."""
        # Get pendulum angle (assuming first joint is hinge)
        angle = self.state.qpos[0] if self.model.nq > 0 else 0
        angular_vel = self.state.qvel[0] if self.model.nv > 0 else 0

        # Reward for being upright
        upright_reward = np.cos(angle)

        # Penalize angular velocity
        vel_penalty = 0.1 * angular_vel ** 2

        # Penalize control effort
        ctrl_penalty = 0.01 * np.sum(action ** 2)

        return upright_reward - vel_penalty - ctrl_penalty

    def _check_termination(self) -> bool:
        """Terminate if pendulum falls too far."""
        if self.model.nq > 0:
            angle = self.state.qpos[0]
            return abs(angle) > np.pi / 2
        return False


class CartPoleEnv(PhysicsEnvironment):
    """Cart-pole balancing task."""

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward for keeping pole upright and cart centered."""
        if self.model.nq < 2:
            return 0.0

        cart_pos = self.state.qpos[0]
        pole_angle = self.state.qpos[1]

        # Reward for pole being upright
        upright_reward = np.cos(pole_angle)

        # Penalty for cart being far from center
        center_penalty = 0.1 * cart_pos ** 2

        # Penalty for control
        ctrl_penalty = 0.01 * np.sum(action ** 2)

        return upright_reward - center_penalty - ctrl_penalty

    def _check_termination(self) -> bool:
        """Terminate if pole falls or cart goes too far."""
        if self.model.nq < 2:
            return False

        cart_pos = self.state.qpos[0]
        pole_angle = self.state.qpos[1]

        cart_limit = 2.4
        angle_limit = np.pi / 4

        return abs(cart_pos) > cart_limit or abs(pole_angle) > angle_limit


class HopperEnv(PhysicsEnvironment):
    """Hopper locomotion task."""

    def __init__(self, model: Model):
        super().__init__(model)
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 0.001
        self._healthy_reward = 1.0

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward for forward movement."""
        # Forward velocity (assuming x is forward direction)
        if self.model.nv >= 3:
            forward_vel = self.state.qvel[0]
        else:
            forward_vel = 0.0

        forward_reward = self._forward_reward_weight * forward_vel
        ctrl_cost = self._ctrl_cost_weight * np.sum(action ** 2)
        healthy_reward = self._healthy_reward if not self._check_termination() else 0.0

        return forward_reward - ctrl_cost + healthy_reward

    def _check_termination(self) -> bool:
        """Terminate if hopper falls."""
        if self.model.nq >= 2:
            height = self.state.qpos[1]  # Assuming y is up
            return height < 0.7
        return False
