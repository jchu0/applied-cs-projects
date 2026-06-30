"""Constraint solver using projected Gauss-Seidel."""

import numpy as np
from typing import List, Tuple, Dict

from ..core.bodies import Model, State, Contact, Joint, JointType


class ConstraintSolver:
    """
    Solve contact constraints using projected Gauss-Seidel.
    Implements LCP (Linear Complementarity Problem) solver.
    """

    def __init__(self, model: Model, iterations: int = 50):
        self.model = model
        self.iterations = iterations
        self.warmstart = True

        # Cache for warmstart
        self._prev_impulses: Dict[Tuple[int, int], float] = {}

    def solve(self, state: State, contacts: List[Contact], dt: float) -> np.ndarray:
        """
        Solve contact constraints and return corrective impulses.

        Args:
            state: Current state
            contacts: Detected contacts
            dt: Timestep

        Returns:
            Corrective impulses for each joint DOF
        """
        if not contacts:
            return np.zeros(self.model.nv)

        n_contacts = len(contacts)

        # Build constraint matrix
        J = np.zeros((n_contacts * 3, self.model.nv))  # Normal + 2 friction
        bias = np.zeros(n_contacts * 3)
        bounds = np.zeros((n_contacts * 3, 2))

        for i, contact in enumerate(contacts):
            # Contact Jacobian
            J_n, J_t1, J_t2 = self._contact_jacobian(state, contact)

            J[i*3, :] = J_n      # Normal
            J[i*3+1, :] = J_t1   # Tangent 1
            J[i*3+2, :] = J_t2   # Tangent 2

            # Bias term (Baumgarte stabilization)
            beta = 0.2  # Stabilization factor
            bias[i*3] = beta * contact.penetration / dt

            # Restitution
            v_rel = J_n @ state.qvel
            if v_rel < -0.5:  # Threshold for restitution
                bias[i*3] += contact.restitution * v_rel

            # Bounds for normal impulse (non-negative)
            bounds[i*3] = (0, np.inf)

            # Friction cone bounds (will be scaled by normal)
            mu = contact.friction
            bounds[i*3+1] = (-mu, mu)
            bounds[i*3+2] = (-mu, mu)

        # Compute effective mass matrix
        M_inv = self._inverse_mass_matrix(state)
        A = J @ M_inv @ J.T

        # Add regularization for stability
        A += np.eye(n_contacts * 3) * 1e-6

        # Solve using projected Gauss-Seidel
        lambda_ = np.zeros(n_contacts * 3)

        # Warmstart
        if self.warmstart:
            for i, contact in enumerate(contacts):
                key = (contact.body1_idx, contact.body2_idx)
                if key in self._prev_impulses:
                    lambda_[i*3] = self._prev_impulses[key]

        b = bias - J @ state.qvel

        for _ in range(self.iterations):
            for i in range(n_contacts * 3):
                # Gauss-Seidel update
                delta = (b[i] - A[i] @ lambda_) / (A[i, i] + 1e-8)
                lambda_new = lambda_[i] + delta

                # Project to constraint bounds
                if i % 3 == 0:
                    # Normal constraint: non-negative
                    lambda_new = max(0, lambda_new)
                else:
                    # Friction constraint: depends on normal
                    contact_idx = i // 3
                    normal_impulse = lambda_[contact_idx * 3]
                    mu = contacts[contact_idx].friction
                    max_friction = mu * normal_impulse
                    lambda_new = np.clip(lambda_new, -max_friction, max_friction)

                lambda_[i] = lambda_new

        # Store for warmstart
        for i, contact in enumerate(contacts):
            key = (contact.body1_idx, contact.body2_idx)
            self._prev_impulses[key] = lambda_[i*3]

        # Convert to joint impulses
        joint_impulse = M_inv @ J.T @ lambda_

        return joint_impulse

    def _contact_jacobian(self, state: State, contact: Contact) -> Tuple[np.ndarray, ...]:
        """
        Compute contact Jacobian mapping joint velocities to contact velocity.
        Returns (J_normal, J_tangent1, J_tangent2)
        """
        nv = self.model.nv

        # Contact frame
        n = contact.normal
        t1 = self._perpendicular(n)
        t2 = np.cross(n, t1)

        J_n = np.zeros(nv)
        J_t1 = np.zeros(nv)
        J_t2 = np.zeros(nv)

        # Contributions from body 1
        r1 = contact.pos - state.xpos[contact.body1_idx]
        body1_J = self._body_jacobian(state, contact.body1_idx)

        # Linear + angular contribution
        for k in range(nv):
            v_lin = body1_J[:3, k]
            v_ang = body1_J[3:, k]
            v_point = v_lin + np.cross(v_ang, r1)

            J_n[k] += np.dot(n, v_point)
            J_t1[k] += np.dot(t1, v_point)
            J_t2[k] += np.dot(t2, v_point)

        # Contributions from body 2 (negative because relative)
        if contact.body2_idx >= 0:
            r2 = contact.pos - state.xpos[contact.body2_idx]
            body2_J = self._body_jacobian(state, contact.body2_idx)

            for k in range(nv):
                v_lin = body2_J[:3, k]
                v_ang = body2_J[3:, k]
                v_point = v_lin + np.cross(v_ang, r2)

                J_n[k] -= np.dot(n, v_point)
                J_t1[k] -= np.dot(t1, v_point)
                J_t2[k] -= np.dot(t2, v_point)

        return J_n, J_t1, J_t2

    def _body_jacobian(self, state: State, body_idx: int) -> np.ndarray:
        """Compute Jacobian mapping joint velocities to body velocity."""
        J = np.zeros((6, self.model.nv))

        # Trace back through kinematic chain
        current = body_idx
        while current >= 0:
            body = self.model.bodies[current]
            for joint in self.model.joints:
                if joint.child_body == current:
                    # Add joint contribution
                    S = self._joint_motion_subspace(joint)
                    start = joint.qvel_idx
                    end = start + joint.n_qvel
                    if end > start:
                        J[:, start:end] += S

            current = body.parent

        return J

    def _inverse_mass_matrix(self, state: State) -> np.ndarray:
        """Compute inverse of mass matrix."""
        # Simplified - use diagonal approximation
        M_inv = np.eye(self.model.nv)

        # Scale by inverse body masses
        for joint in self.model.joints:
            if joint.child_body >= 0:
                body = self.model.bodies[joint.child_body]
                m = body.inertia.mass
                if m > 0:
                    start = joint.qvel_idx
                    end = start + joint.n_qvel
                    M_inv[start:end, start:end] /= m

        return M_inv

    def _perpendicular(self, v: np.ndarray) -> np.ndarray:
        """Get a vector perpendicular to v."""
        v = v / (np.linalg.norm(v) + 1e-8)
        if abs(v[0]) < 0.9:
            p = np.array([1.0, 0.0, 0.0])
        else:
            p = np.array([0.0, 1.0, 0.0])
        perp = np.cross(v, p)
        return perp / (np.linalg.norm(perp) + 1e-8)

    def _joint_motion_subspace(self, joint: Joint) -> np.ndarray:
        """Get motion subspace matrix."""
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
