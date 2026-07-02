"""Tests for the Gym-style environment wrapper.

Focus on reproducibility: each environment owns a private RNG, so seeding one
environment must not perturb another, and identical seeds must yield identical
trajectories.
"""

import copy

import numpy as np
import pytest

from physicsrl.environment.gym_env import PhysicsEnvironment, BatchedEnvironment


def _fresh_model(cart_pole_model):
    """Deep-copy the shared model fixture so per-env mass/friction scaling in
    one test's environments does not leak into another environment."""
    return copy.deepcopy(cart_pole_model)


def _rollout(env, n_steps=25):
    """Run a fixed zero-action rollout and collect observations."""
    obs = [env.reset(seed=None)]
    action = np.zeros(env.action_dim)
    for _ in range(n_steps):
        o, _, _, _, _ = env.step(action)
        obs.append(o)
    return np.array(obs)


class TestReproducibility:
    """Reproducibility and RNG-isolation regression tests."""

    def test_same_seed_same_trajectory(self, cart_pole_model):
        """Two envs reset with the same seed produce identical trajectories."""
        env_a = PhysicsEnvironment(_fresh_model(cart_pole_model))
        env_b = PhysicsEnvironment(_fresh_model(cart_pole_model))

        obs_a = env_a.reset(seed=1234)
        obs_b = env_b.reset(seed=1234)
        np.testing.assert_array_equal(obs_a, obs_b)

        traj_a = _rollout(env_a)
        traj_b = _rollout(env_b)
        np.testing.assert_allclose(traj_a, traj_b, rtol=0, atol=0)

    def test_different_seed_different_randomization(self, cart_pole_model):
        """Different seeds yield different domain randomization draws."""
        env_a = PhysicsEnvironment(_fresh_model(cart_pole_model))
        env_b = PhysicsEnvironment(_fresh_model(cart_pole_model))

        env_a.reset(seed=1)
        env_b.reset(seed=2)

        masses_a = [b.inertia.mass for b in env_a.model.bodies]
        masses_b = [b.inertia.mass for b in env_b.model.bodies]
        assert masses_a != masses_b

    def test_seeding_one_env_does_not_perturb_another(self, cart_pole_model):
        """Seeding/sampling in a third env must not change a seeded env's output.

        This is the core regression: with the old global-RNG implementation,
        calling reset(seed=) or sampling on any environment reseeded the shared
        process-global RNG and silently changed other environments' results.
        """
        # Reference environment, seeded and rolled out in isolation.
        ref = PhysicsEnvironment(_fresh_model(cart_pole_model))
        ref.reset(seed=42)
        ref_traj = _rollout(ref)

        # Now seed the same env again, but interleave a THIRD env that seeds
        # and samples heavily in between. If RNGs were global, the third env's
        # activity would corrupt the target env's trajectory.
        target = PhysicsEnvironment(_fresh_model(cart_pole_model))
        target.reset(seed=42)

        noise = PhysicsEnvironment(_fresh_model(cart_pole_model))
        for s in range(10):
            noise.reset(seed=s)
            noise.rng.uniform(size=100)

        target_traj = _rollout(target)
        np.testing.assert_allclose(ref_traj, target_traj, rtol=0, atol=0)

    def test_reset_does_not_touch_global_rng(self, cart_pole_model):
        """reset(seed=) must not disturb the process-global NumPy RNG."""
        np.random.seed(7)
        before = np.random.random()

        np.random.seed(7)
        env = PhysicsEnvironment(_fresh_model(cart_pole_model))
        env.reset(seed=99999)
        after = np.random.random()

        # Global RNG stream is unaffected by the env's private seeding.
        assert before == after

    def test_env_owns_generator(self, cart_pole_model):
        """Environment exposes a per-instance numpy Generator, not global RNG."""
        env = PhysicsEnvironment(_fresh_model(cart_pole_model))
        assert isinstance(env.rng, np.random.Generator)


class TestBatchedReproducibility:
    """Reproducibility for the batched environment wrapper."""

    def test_batched_seed_reproducible(self, cart_pole_model):
        """Two batched envs built with the same seed match element-wise."""
        batch_a = BatchedEnvironment(_fresh_model(cart_pole_model), num_envs=3, seed=5)
        batch_b = BatchedEnvironment(_fresh_model(cart_pole_model), num_envs=3, seed=5)

        obs_a = batch_a.reset(seed=5)
        obs_b = batch_b.reset(seed=5)
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_batched_sub_envs_independent(self, cart_pole_model):
        """Each sub-env has its own Generator instance."""
        batch = BatchedEnvironment(_fresh_model(cart_pole_model), num_envs=3, seed=5)
        rngs = [env.rng for env in batch.envs]
        # Distinct Generator objects.
        assert len({id(r) for r in rngs}) == 3
