"""Forward dynamics using Articulated Body Algorithm."""

import numpy as np
from typing import Optional

from ..core.bodies import (
    Model, State, Body, Joint, JointType,
    quat_mul, quat_rotate, axis_angle_to_quat
)


class ForwardDynamics:
    """
    Computes forward dynamics: accelerations from forces/torques.
    Implements Articulated Body Algorithm (ABA).
    """

    def __init__(self, model: Model):
        self.model = model

        # Pre-allocate workspace
        self.spatial_inertia = np.zeros((model.nbody, 6, 6))
        self.bias_force = np.zeros((model.nbody, 6))
        self.spatial_transform = np.zeros((model.nbody, 6, 6))
        self.body_acc = np.zeros((model.nbody, 6))

    def compute(self, state: State, tau: np.ndarray) -> np.ndarray:
        """
        Compute joint accelerations from applied torques.

        Args:
            state: Current state
            tau: Applied joint torques (nv,)

        Returns:
            qacc: Joint accelerations (nv,)
        """
        model = self.model

        # Step 1: Forward pass - compute transforms and velocities
        self._forward_kinematics(state)

        # Step 2: Backward pass - compute articulated body inertias
        for i in reversed(range(model.nbody)):
            body = model.bodies[i]

            # Initialize with rigid body inertia
            self.spatial_inertia[i] = self._body_spatial_inertia(body)

            # Compute bias force (gravity + Coriolis)
            self.bias_force[i] = self._compute_bias_force(state, i)

            # Add children contributions
            for j in range(model.nbody):
                if model.bodies[j].parent == i:
                    # Transform child inertia to this frame
                    child_inertia = self._transform_inertia(
                        self.spatial_inertia[j],
                        self.spatial_transform[j]
                    )
                    self.spatial_inertia[i] += child_inertia

        # Step 3: Forward pass - compute accelerations
        qacc = np.zeros(model.nv)

        # Base acceleration (gravity)
        a0 = np.concatenate([np.zeros(3), -model.gravity])

        for i in range(model.nbody):
            body = model.bodies[i]

            # Get parent acceleration
            if body.parent == -1:
                a_parent = a0
            else:
                a_parent = self.body_acc[body.parent]

            # Find joint for this body
            joint = None
            for j in self.model.joints:
                if j.child_body == i:
                    joint = j
                    break

            if joint is None:
                self.body_acc[i] = a_parent
                continue

            # Compute joint acceleration
            S = self._joint_motion_subspace(joint)
            c = self._velocity_product(state, i)

            if S.shape[1] == 0:
                # Fixed joint
                self.body_acc[i] = a_parent
                continue

            # Articulated body equation
            H = S.T @ self.spatial_inertia[i] @ S
            h = S.T @ (self.spatial_inertia[i] @ a_parent + self.bias_force[i])

            joint_tau = tau[joint.qvel_idx:joint.qvel_idx + joint.n_qvel]

            # Add damping
            if joint.damping > 0:
                joint_vel = state.qvel[joint.qvel_idx:joint.qvel_idx + joint.n_qvel]
                joint_tau = joint_tau - joint.damping * joint_vel

            # Solve for acceleration
            if H.size == 1:
                joint_qacc = (joint_tau - h) / (H + 1e-8)
            else:
                joint_qacc = np.linalg.solve(H + np.eye(H.shape[0]) * 1e-8, joint_tau - h)

            qacc[joint.qvel_idx:joint.qvel_idx + joint.n_qvel] = joint_qacc.flatten()

            # Compute body acceleration
            self.body_acc[i] = a_parent + S @ joint_qacc.reshape(-1, 1).flatten() + c

        return qacc

    def _forward_kinematics(self, state: State):
        """Compute body positions and velocities from joint state."""
        model = self.model

        for i in range(model.nbody):
            body = model.bodies[i]

            if body.parent == -1:
                # Root body - find free joint
                for joint in model.joints:
                    if joint.child_body == i and joint.joint_type == JointType.FREE:
                        state.xpos[i] = state.qpos[joint.qpos_idx:joint.qpos_idx+3]
                        state.xquat[i] = state.qpos[joint.qpos_idx+3:joint.qpos_idx+7]
                        # Normalize quaternion
                        state.xquat[i] /= np.linalg.norm(state.xquat[i])
                        break
                else:
                    state.xpos[i] = body.pos
                    state.xquat[i] = body.quat
            else:
                # Find joint connecting to parent
                for joint in model.joints:
                    if joint.child_body == i:
                        # Compute transform based on joint type
                        parent_pos = state.xpos[body.parent]
                        parent_quat = state.xquat[body.parent]

                        if joint.joint_type == JointType.HINGE:
                            angle = state.qpos[joint.qpos_idx]
                            q = axis_angle_to_quat(joint.axis, angle)
                            state.xquat[i] = quat_mul(parent_quat, q)
                            state.xpos[i] = parent_pos + quat_rotate(parent_quat, joint.pos)

                        elif joint.joint_type == JointType.SLIDE:
                            offset = joint.axis * state.qpos[joint.qpos_idx]
                            state.xquat[i] = parent_quat.copy()
                            state.xpos[i] = parent_pos + quat_rotate(
                                parent_quat, joint.pos + offset
                            )

                        elif joint.joint_type == JointType.BALL:
                            q = state.qpos[joint.qpos_idx:joint.qpos_idx+4]
                            q = q / np.linalg.norm(q)
                            state.xquat[i] = quat_mul(parent_quat, q)
                            state.xpos[i] = parent_pos + quat_rotate(parent_quat, joint.pos)

                        elif joint.joint_type == JointType.FIXED:
                            state.xquat[i] = parent_quat.copy()
                            state.xpos[i] = parent_pos + quat_rotate(parent_quat, joint.pos)

                        # Compute spatial transform
                        self.spatial_transform[i] = self._compute_spatial_transform(
                            state.xpos[body.parent], state.xquat[body.parent],
                            state.xpos[i], state.xquat[i]
                        )
                        break

    def _body_spatial_inertia(self, body: Body) -> np.ndarray:
        """Compute 6x6 spatial inertia matrix for body."""
        m = body.inertia.mass
        I = body.inertia.inertia
        c = body.inertia.com

        # Spatial inertia in body frame
        c_cross = self._skew(c)

        spatial = np.zeros((6, 6))
        spatial[:3, :3] = I - m * c_cross @ c_cross
        spatial[:3, 3:] = m * c_cross
        spatial[3:, :3] = m * c_cross.T
        spatial[3:, 3:] = m * np.eye(3)

        return spatial

    def _compute_bias_force(self, state: State, body_idx: int) -> np.ndarray:
        """Compute bias force (gravity + Coriolis) for body."""
        body = self.model.bodies[body_idx]
        m = body.inertia.mass

        # Gravity force in spatial coordinates
        R = self._quat_to_rot(state.xquat[body_idx])
        g_body = R.T @ self.model.gravity

        bias = np.zeros(6)
        bias[3:] = -m * g_body  # Gravity contribution

        return bias

    def _joint_motion_subspace(self, joint: Joint) -> np.ndarray:
        """Get motion subspace matrix for joint."""
        if joint.joint_type == JointType.HINGE:
            S = np.zeros((6, 1))
            S[:3, 0] = joint.axis
            return S

        elif joint.joint_type == JointType.SLIDE:
            S = np.zeros((6, 1))
            S[3:, 0] = joint.axis
            return S

        elif joint.joint_type == JointType.BALL:
            return np.vstack([np.eye(3), np.zeros((3, 3))])

        elif joint.joint_type == JointType.FREE:
            return np.eye(6)

        else:
            return np.zeros((6, 0))

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    def _velocity_product(self, state: State, body_idx: int) -> np.ndarray:
        """Compute velocity-dependent acceleration term."""
        # Coriolis and centrifugal terms
        return np.zeros(6)  # Simplified

    def _transform_inertia(self, I: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Transform spatial inertia by spatial transform."""
        return X.T @ I @ X

    def _compute_spatial_transform(self, pos1, quat1, pos2, quat2) -> np.ndarray:
        """Compute 6x6 spatial transform between frames."""
        # Relative rotation
        R1 = self._quat_to_rot(quat1)
        R2 = self._quat_to_rot(quat2)
        R = R2.T @ R1

        # Relative translation
        p = R2.T @ (pos1 - pos2)
        p_cross = self._skew(p)

        # 6x6 spatial transform
        X = np.zeros((6, 6))
        X[:3, :3] = R
        X[3:, 3:] = R
        X[3:, :3] = p_cross @ R

        return X
