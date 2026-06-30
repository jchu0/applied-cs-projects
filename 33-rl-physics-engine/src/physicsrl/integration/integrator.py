"""Numerical integrators for physics simulation."""

import numpy as np

from ..core.bodies import Model, State, JointType
from ..dynamics.forward import ForwardDynamics
from ..collision.detection import CollisionSystem
from ..solver.constraints import ConstraintSolver


class Integrator:
    """Numerical integrator for time stepping."""

    def __init__(self, model: Model):
        self.model = model
        self.dynamics = ForwardDynamics(model)
        self.collision = CollisionSystem(model)
        self.constraint = ConstraintSolver(model)

    def step(self, state: State, ctrl: np.ndarray) -> State:
        """
        Advance simulation by one timestep.

        Args:
            state: Current state
            ctrl: Control inputs

        Returns:
            New state after timestep
        """
        dt = self.model.timestep
        state.ctrl = ctrl

        # Compute applied forces from actuators
        tau = self._compute_actuator_forces(state)

        if self.model.integrator == "euler":
            new_state = self._euler_step(state, tau, dt)

        elif self.model.integrator == "rk4":
            new_state = self._rk4_step(state, tau, dt)

        elif self.model.integrator == "semi_implicit":
            new_state = self._semi_implicit_step(state, tau, dt)

        else:
            raise ValueError(f"Unknown integrator: {self.model.integrator}")

        # Detect contacts
        contacts = self.collision.detect_contacts(new_state)
        new_state.contacts = contacts

        # Solve constraints
        if contacts:
            impulse = self.constraint.solve(new_state, contacts, dt)
            new_state.qvel += impulse

        # Update time
        new_state.time = state.time + dt

        return new_state

    def _euler_step(self, state: State, tau: np.ndarray, dt: float) -> State:
        """Simple Euler integration."""
        new_state = State.create(self.model)

        # Compute accelerations
        qacc = self.dynamics.compute(state, tau)

        # Update velocities
        new_state.qvel = state.qvel + qacc * dt

        # Update positions
        new_state.qpos = state.qpos + state.qvel * dt

        # Handle quaternions
        self._normalize_quaternions(new_state)

        # Update forward kinematics
        self.dynamics._forward_kinematics(new_state)

        return new_state

    def _semi_implicit_step(self, state: State, tau: np.ndarray, dt: float) -> State:
        """Semi-implicit Euler (symplectic)."""
        new_state = State.create(self.model)

        # Compute accelerations
        qacc = self.dynamics.compute(state, tau)

        # Update velocities first
        new_state.qvel = state.qvel + qacc * dt

        # Apply joint damping
        for joint in self.model.joints:
            if joint.damping > 0:
                start = joint.qvel_idx
                end = start + joint.n_qvel
                new_state.qvel[start:end] *= (1 - joint.damping * dt)

        # Update positions using new velocity
        new_state.qpos = state.qpos + new_state.qvel * dt

        # Apply joint limits
        self._apply_joint_limits(new_state)

        self._normalize_quaternions(new_state)
        self.dynamics._forward_kinematics(new_state)

        return new_state

    def _rk4_step(self, state: State, tau: np.ndarray, dt: float) -> State:
        """Fourth-order Runge-Kutta integration."""
        def f(q, v):
            """Compute derivatives."""
            temp_state = State.create(self.model)
            temp_state.qpos = q
            temp_state.qvel = v
            self.dynamics._forward_kinematics(temp_state)
            qacc = self.dynamics.compute(temp_state, tau)
            return v, qacc

        q0, v0 = state.qpos.copy(), state.qvel.copy()

        # k1
        dq1, dv1 = f(q0, v0)

        # k2
        dq2, dv2 = f(q0 + 0.5*dt*dq1, v0 + 0.5*dt*dv1)

        # k3
        dq3, dv3 = f(q0 + 0.5*dt*dq2, v0 + 0.5*dt*dv2)

        # k4
        dq4, dv4 = f(q0 + dt*dq3, v0 + dt*dv3)

        new_state = State.create(self.model)
        new_state.qpos = q0 + dt * (dq1 + 2*dq2 + 2*dq3 + dq4) / 6
        new_state.qvel = v0 + dt * (dv1 + 2*dv2 + 2*dv3 + dv4) / 6

        self._normalize_quaternions(new_state)
        self.dynamics._forward_kinematics(new_state)

        return new_state

    def _compute_actuator_forces(self, state: State) -> np.ndarray:
        """Convert control inputs to joint forces/torques."""
        tau = np.zeros(self.model.nv)

        for i, actuator in enumerate(self.model.actuators):
            joint = self.model.joints[actuator.joint_idx]

            # Clip control to range
            ctrl = np.clip(state.ctrl[i], *actuator.ctrl_range)

            # Apply gear ratio
            force = ctrl * actuator.gear

            # Clip to force range
            force = np.clip(force, *actuator.forcerange)

            # Add to joint torque
            tau[joint.qvel_idx:joint.qvel_idx + joint.n_qvel] += force

        return tau

    def _normalize_quaternions(self, state: State):
        """Normalize all quaternions in state."""
        # For free and ball joints
        for joint in self.model.joints:
            if joint.joint_type == JointType.FREE:
                q_idx = joint.qpos_idx + 3  # Skip position
                q = state.qpos[q_idx:q_idx+4]
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    state.qpos[q_idx:q_idx+4] = q / norm
                else:
                    state.qpos[q_idx:q_idx+4] = np.array([1, 0, 0, 0])

            elif joint.joint_type == JointType.BALL:
                q_idx = joint.qpos_idx
                q = state.qpos[q_idx:q_idx+4]
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    state.qpos[q_idx:q_idx+4] = q / norm
                else:
                    state.qpos[q_idx:q_idx+4] = np.array([1, 0, 0, 0])

    def _apply_joint_limits(self, state: State):
        """Apply joint position limits."""
        for joint in self.model.joints:
            if joint.joint_type in {JointType.HINGE, JointType.SLIDE}:
                idx = joint.qpos_idx
                pos = state.qpos[idx]

                if pos < joint.limit_lower:
                    state.qpos[idx] = joint.limit_lower
                    state.qvel[joint.qvel_idx] = max(0, state.qvel[joint.qvel_idx])

                elif pos > joint.limit_upper:
                    state.qpos[idx] = joint.limit_upper
                    state.qvel[joint.qvel_idx] = min(0, state.qvel[joint.qvel_idx])
