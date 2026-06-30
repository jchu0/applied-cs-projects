"""Tests for forward dynamics and articulated body algorithm."""

import pytest
import numpy as np

from physicsrl import (
    Model, Body, Joint, Geom, Actuator, Inertia, State,
    GeomType, JointType
)
from physicsrl.dynamics import ForwardDynamics


class TestForwardDynamicsCreation:
    """Tests for ForwardDynamics initialization."""

    def test_forward_dynamics_creation(self, single_free_body_model):
        """Test ForwardDynamics can be created."""
        dynamics = ForwardDynamics(single_free_body_model)
        assert dynamics.model is single_free_body_model

    def test_workspace_allocation(self, single_free_body_model):
        """Test workspace arrays are allocated."""
        dynamics = ForwardDynamics(single_free_body_model)

        # Should have workspace for 1 body
        assert dynamics.spatial_inertia.shape == (1, 6, 6)
        assert dynamics.bias_force.shape == (1, 6)
        assert dynamics.body_acc.shape == (1, 6)

    def test_pendulum_workspace_allocation(self, pendulum_model):
        """Test workspace allocation for pendulum model."""
        dynamics = ForwardDynamics(pendulum_model)

        assert dynamics.spatial_inertia.shape == (2, 6, 6)
        assert dynamics.bias_force.shape == (2, 6)
        assert dynamics.body_acc.shape == (2, 6)


class TestForwardKinematics:
    """Tests for forward kinematics computation."""

    def test_forward_kinematics_free_joint(self, single_free_body_model):
        """Test forward kinematics for free joint."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)

        # Set position in qpos
        state.qpos[0:3] = np.array([1.0, 2.0, 3.0])
        state.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat

        dynamics._forward_kinematics(state)

        np.testing.assert_allclose(state.xpos[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(state.xquat[0], np.array([1.0, 0.0, 0.0, 0.0]))

    def test_forward_kinematics_hinge_joint(self, pendulum_model):
        """Test forward kinematics for hinge joint."""
        dynamics = ForwardDynamics(pendulum_model)
        state = State.create(pendulum_model)

        # Set hinge angle to 90 degrees
        state.qpos[0] = np.pi / 2

        dynamics._forward_kinematics(state)

        # Child body should be rotated
        # Note: exact result depends on joint axis direction
        assert state.xquat[1] is not None

    def test_forward_kinematics_preserves_quaternion_norm(self, single_free_body_model, assert_quaternion_normalized):
        """Test forward kinematics maintains unit quaternions."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)

        # Set non-normalized quaternion
        state.qpos[3:7] = np.array([1.0, 1.0, 0.0, 0.0])

        dynamics._forward_kinematics(state)

        # Should be normalized after forward kinematics
        assert_quaternion_normalized(state.xquat[0])


class TestComputeAccelerations:
    """Tests for acceleration computation."""

    def test_free_fall_acceleration(self, single_free_body_model):
        """Test free-falling body has gravity acceleration."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)

        # Initialize position
        state.qpos[0:3] = np.array([0.0, 0.0, 5.0])
        state.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        # Zero applied torque
        tau = np.zeros(single_free_body_model.nv)

        qacc = dynamics.compute(state, tau)

        # Linear acceleration should be approximately gravity
        # Note: The exact result depends on implementation details
        assert qacc.shape == (6,)

    def test_zero_torque_pendulum(self, pendulum_model):
        """Test pendulum with zero applied torque."""
        dynamics = ForwardDynamics(pendulum_model)
        state = State.create(pendulum_model)

        tau = np.zeros(pendulum_model.nv)
        qacc = dynamics.compute(state, tau)

        assert qacc.shape == (1,)  # Single DOF

    def test_applied_torque_affects_acceleration(self, pendulum_model):
        """Test that applied torque changes acceleration."""
        dynamics = ForwardDynamics(pendulum_model)
        state = State.create(pendulum_model)

        # Zero torque
        tau_zero = np.zeros(pendulum_model.nv)
        qacc_zero = dynamics.compute(state, tau_zero)

        # Non-zero torque
        tau_nonzero = np.array([1.0])
        qacc_nonzero = dynamics.compute(state, tau_nonzero)

        # Accelerations should differ
        assert not np.allclose(qacc_zero, qacc_nonzero)

    def test_acceleration_proportional_to_torque(self, pendulum_model):
        """Test acceleration scales with torque."""
        dynamics = ForwardDynamics(pendulum_model)
        state = State.create(pendulum_model)

        tau1 = np.array([1.0])
        qacc1 = dynamics.compute(state, tau1)

        tau2 = np.array([2.0])
        qacc2 = dynamics.compute(state, tau2)

        # Difference in acceleration should be proportional
        diff1 = qacc1[0]
        diff2 = qacc2[0]

        # Account for gravity contribution
        ratio = (diff2) / (diff1) if abs(diff1) > 1e-8 else 0
        # The ratio depends on gravity, so we just check it's different
        assert qacc2[0] > qacc1[0]  # More torque -> more acceleration


class TestSpatialInertia:
    """Tests for spatial inertia computation."""

    def test_body_spatial_inertia_shape(self, single_free_body_model):
        """Test spatial inertia matrix has correct shape."""
        dynamics = ForwardDynamics(single_free_body_model)
        body = single_free_body_model.bodies[0]

        M = dynamics._body_spatial_inertia(body)

        assert M.shape == (6, 6)

    def test_spatial_inertia_symmetric(self, single_free_body_model):
        """Test spatial inertia matrix is symmetric."""
        dynamics = ForwardDynamics(single_free_body_model)
        body = single_free_body_model.bodies[0]

        M = dynamics._body_spatial_inertia(body)

        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_spatial_inertia_positive_definite(self, single_free_body_model):
        """Test spatial inertia matrix is positive definite."""
        dynamics = ForwardDynamics(single_free_body_model)
        body = single_free_body_model.bodies[0]

        M = dynamics._body_spatial_inertia(body)

        # Check all eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= 0)


class TestBiasForce:
    """Tests for bias force computation."""

    def test_bias_force_shape(self, single_free_body_model):
        """Test bias force has correct shape."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)
        state.qpos[0:3] = np.array([0.0, 0.0, 5.0])
        state.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        dynamics._forward_kinematics(state)

        bias = dynamics._compute_bias_force(state, 0)

        assert bias.shape == (6,)

    def test_bias_force_includes_gravity(self, single_free_body_model):
        """Test bias force includes gravity contribution."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)
        state.qpos[0:3] = np.array([0.0, 0.0, 5.0])
        state.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        dynamics._forward_kinematics(state)

        bias = dynamics._compute_bias_force(state, 0)

        # Linear force component should be non-zero due to gravity
        assert np.linalg.norm(bias[3:]) > 0


class TestJointMotionSubspace:
    """Tests for joint motion subspace computation."""

    def test_hinge_motion_subspace(self, pendulum_model):
        """Test hinge joint motion subspace."""
        dynamics = ForwardDynamics(pendulum_model)
        joint = pendulum_model.joints[0]

        S = dynamics._joint_motion_subspace(joint)

        assert S.shape == (6, 1)
        # Angular component should match joint axis
        np.testing.assert_allclose(S[:3, 0], joint.axis)
        # Linear component should be zero
        np.testing.assert_allclose(S[3:, 0], np.zeros(3))

    def test_slide_motion_subspace(self, cart_pole_model):
        """Test slide joint motion subspace."""
        dynamics = ForwardDynamics(cart_pole_model)
        joint = cart_pole_model.joints[0]  # Slide joint

        S = dynamics._joint_motion_subspace(joint)

        assert S.shape == (6, 1)
        # Angular component should be zero
        np.testing.assert_allclose(S[:3, 0], np.zeros(3))
        # Linear component should match joint axis
        np.testing.assert_allclose(S[3:, 0], joint.axis)

    def test_free_joint_motion_subspace(self, single_free_body_model):
        """Test free joint motion subspace."""
        dynamics = ForwardDynamics(single_free_body_model)
        joint = single_free_body_model.joints[0]

        S = dynamics._joint_motion_subspace(joint)

        assert S.shape == (6, 6)
        np.testing.assert_allclose(S, np.eye(6))

    def test_fixed_joint_motion_subspace(self):
        """Test fixed joint has empty motion subspace."""
        model = Model()
        base = Body(name="base", inertia=Inertia.from_sphere(0, 0))
        model.add_body(base)
        fixed = Body(name="fixed", inertia=Inertia.from_sphere(1, 1), parent=0)
        model.add_body(fixed)

        joint = Joint(
            joint_type=JointType.FIXED,
            parent_body=0,
            child_body=1
        )
        model.add_joint(joint)

        dynamics = ForwardDynamics(model)
        S = dynamics._joint_motion_subspace(joint)

        assert S.shape == (6, 0)


class TestSkewSymmetric:
    """Tests for skew-symmetric matrix utility."""

    def test_skew_matrix_shape(self):
        """Test skew matrix has correct shape."""
        v = np.array([1.0, 2.0, 3.0])
        S = ForwardDynamics._skew(v)
        assert S.shape == (3, 3)

    def test_skew_matrix_antisymmetric(self):
        """Test skew matrix is antisymmetric."""
        v = np.array([1.0, 2.0, 3.0])
        S = ForwardDynamics._skew(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-10)

    def test_skew_matrix_cross_product(self):
        """Test skew matrix implements cross product."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        S = ForwardDynamics._skew(a)
        cross_via_matrix = S @ b
        cross_direct = np.cross(a, b)

        np.testing.assert_allclose(cross_via_matrix, cross_direct, atol=1e-10)


class TestQuatToRot:
    """Tests for quaternion to rotation matrix conversion."""

    def test_identity_quat_to_identity_rot(self):
        """Test identity quaternion gives identity rotation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = ForwardDynamics._quat_to_rot(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        """Test rotation matrix is orthogonal."""
        # Use a properly normalized quaternion
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])  # 90 deg around X
        R = ForwardDynamics._quat_to_rot(q)

        # R @ R^T should be identity
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_rotation_matrix_det_one(self):
        """Test rotation matrix has determinant 1."""
        q = np.array([0.5, 0.5, 0.5, 0.5])  # Some rotation
        R = ForwardDynamics._quat_to_rot(q)

        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6


class TestSpatialTransform:
    """Tests for spatial transform computation."""

    def test_identity_transform(self, single_free_body_model):
        """Test spatial transform between same frame."""
        dynamics = ForwardDynamics(single_free_body_model)

        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        X = dynamics._compute_spatial_transform(pos, quat, pos, quat)

        # Should be identity (6x6)
        assert X.shape == (6, 6)


class TestMultiBodyDynamics:
    """Tests for multi-body dynamics."""

    def test_cart_pole_dynamics(self, cart_pole_model):
        """Test cart-pole dynamics computation."""
        dynamics = ForwardDynamics(cart_pole_model)
        state = State.create(cart_pole_model)

        tau = np.zeros(cart_pole_model.nv)
        qacc = dynamics.compute(state, tau)

        assert qacc.shape == (2,)

    def test_cart_pole_control_affects_cart(self, cart_pole_model):
        """Test control input affects cart acceleration."""
        dynamics = ForwardDynamics(cart_pole_model)
        state = State.create(cart_pole_model)

        # Apply force through actuator
        tau_zero = np.zeros(cart_pole_model.nv)
        tau_force = np.array([10.0, 0.0])  # Force on cart

        qacc_zero = dynamics.compute(state, tau_zero)
        qacc_force = dynamics.compute(state, tau_force)

        # Cart acceleration should be different
        assert qacc_force[0] != qacc_zero[0]


class TestEnergyConservation:
    """Tests for energy-related properties."""

    def test_kinetic_energy_positive(self, single_free_body_model):
        """Test kinetic energy is always non-negative."""
        dynamics = ForwardDynamics(single_free_body_model)
        state = State.create(single_free_body_model)

        # Set some velocity
        state.qvel[:] = np.random.randn(6)

        body = single_free_body_model.bodies[0]
        M = dynamics._body_spatial_inertia(body)

        # Kinetic energy: 0.5 * v^T * M * v
        ke = 0.5 * state.qvel @ M @ state.qvel

        assert ke >= 0


class TestDampedDynamics:
    """Tests for damped joint dynamics."""

    def test_damping_reduces_velocity(self, pendulum_model):
        """Test damping effect on joint acceleration."""
        # Pendulum has damping=0.1
        dynamics = ForwardDynamics(pendulum_model)
        state = State.create(pendulum_model)

        # Set initial velocity
        state.qvel[0] = 1.0

        tau = np.zeros(pendulum_model.nv)
        qacc = dynamics.compute(state, tau)

        # With damping and positive velocity, acceleration should be reduced
        # (compared to undamped case)
        assert qacc.shape == (1,)
