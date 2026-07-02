"""Numerical integrators for physics simulation."""

import numpy as np

from ..core.bodies import Model, State, JointType, quat_mul
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

        # Update positions (per-joint; quaternions integrate via angular velocity)
        new_state.qpos = self._integrate_positions(state.qpos, state.qvel, dt)

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

        # Update positions using new velocity (per-joint integration)
        new_state.qpos = self._integrate_positions(state.qpos, new_state.qvel, dt)

        # Apply joint limits
        self._apply_joint_limits(new_state)

        self._normalize_quaternions(new_state)
        self.dynamics._forward_kinematics(new_state)

        return new_state

    def _rk4_step(self, state: State, tau: np.ndarray, dt: float) -> State:
        """Fourth-order Runge-Kutta integration."""
        def accel(q, v):
            """Compute acceleration at a given position/velocity."""
            temp_state = State.create(self.model)
            temp_state.qpos = q
            temp_state.qvel = v
            self.dynamics._forward_kinematics(temp_state)
            return self.dynamics.compute(temp_state, tau)

        q0, v0 = state.qpos.copy(), state.qvel.copy()

        # RK4 on velocity (position derivative is velocity itself). Positions at
        # intermediate stages are advanced with the per-joint integrator so that
        # quaternion DOFs stay on the manifold instead of being blended linearly.
        a1 = accel(q0, v0)

        v2 = v0 + 0.5*dt*a1
        q2 = self._integrate_positions(q0, v0, 0.5*dt)
        a2 = accel(q2, v2)

        v3 = v0 + 0.5*dt*a2
        q3 = self._integrate_positions(q0, v2, 0.5*dt)
        a3 = accel(q3, v3)

        v4 = v0 + dt*a3
        q4 = self._integrate_positions(q0, v3, dt)
        a4 = accel(q4, v4)

        new_state = State.create(self.model)
        new_state.qvel = v0 + dt * (a1 + 2*a2 + 2*a3 + a4) / 6
        # Advance positions with the RK4-averaged velocity, per joint type.
        v_avg = (v0 + 2*v2 + 2*v3 + v4) / 6
        new_state.qpos = self._integrate_positions(q0, v_avg, dt)

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

    def _integrate_positions(self, qpos: np.ndarray, qvel: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Advance generalized positions by one timestep, per joint type.

        Translational DOFs integrate linearly (pos += lin_vel * dt), while
        orientation quaternions integrate through the angular velocity using the
        quaternion derivative  q_dot = 0.5 * quat(0, omega) * q, giving
        q_next = normalize(q + q_dot * dt). This is required because qpos and
        qvel have different sizes for FREE (7 vs 6) and BALL (4 vs 3) joints,
        so a single flat  qpos + qvel * dt  is dimensionally invalid.

        HINGE and SLIDE joints are 1:1 and reduce exactly to  q + v * dt,
        preserving their previous behavior.
        """
        new_qpos = qpos.copy()

        for joint in self.model.joints:
            pi = joint.qpos_idx
            vi = joint.qvel_idx

            if joint.joint_type == JointType.FREE:
                # 3 translational DOFs + quaternion (7 qpos over 6 qvel).
                lin_vel = qvel[vi:vi + 3]
                omega = qvel[vi + 3:vi + 6]
                new_qpos[pi:pi + 3] = qpos[pi:pi + 3] + lin_vel * dt

                q = qpos[pi + 3:pi + 7]
                new_qpos[pi + 3:pi + 7] = self._integrate_quat(q, omega, dt)

            elif joint.joint_type == JointType.BALL:
                # Quaternion orientation only (4 qpos over 3 qvel).
                omega = qvel[vi:vi + 3]
                q = qpos[pi:pi + 4]
                new_qpos[pi:pi + 4] = self._integrate_quat(q, omega, dt)

            elif joint.joint_type in {JointType.HINGE, JointType.SLIDE}:
                new_qpos[pi] = qpos[pi] + qvel[vi] * dt

            # FIXED: no DOFs, nothing to integrate.

        return new_qpos

    @staticmethod
    def _integrate_quat(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate a unit quaternion by angular velocity omega over dt.

        Uses the quaternion derivative  q_dot = 0.5 * quat(0, omega) * q  and a
        single explicit step  q_next = normalize(q + q_dot * dt).
        """
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_mul(omega_quat, q)
        q_next = q + q_dot * dt
        norm = np.linalg.norm(q_next)
        if norm > 1e-8:
            return q_next / norm
        return np.array([1.0, 0.0, 0.0, 0.0])

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
