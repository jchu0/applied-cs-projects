"""Tests for rigid body data structures and core physics components."""

import pytest
import numpy as np

from physicsrl import (
    Model, Body, Joint, Geom, Actuator, Inertia, State, Contact,
    GeomType, JointType
)
from physicsrl.core.bodies import (
    quat_mul, quat_rotate, quat_conj, axis_angle_to_quat, quat_to_mat
)


class TestInertia:
    """Tests for Inertia class."""

    def test_from_sphere_creates_valid_inertia(self):
        """Test sphere inertia computation."""
        mass = 2.0
        radius = 0.5
        inertia = Inertia.from_sphere(mass, radius)

        assert inertia.mass == mass
        np.testing.assert_array_equal(inertia.com, np.zeros(3))

        # Sphere inertia: I = (2/5) * m * r^2
        expected_i = 2 * mass * radius * radius / 5
        expected_tensor = np.eye(3) * expected_i
        np.testing.assert_allclose(inertia.inertia, expected_tensor)

    def test_from_box_creates_valid_inertia(self):
        """Test box inertia computation."""
        mass = 3.0
        half_extents = np.array([0.5, 1.0, 1.5])
        inertia = Inertia.from_box(mass, half_extents)

        assert inertia.mass == mass
        np.testing.assert_array_equal(inertia.com, np.zeros(3))

        # Box inertia formula
        x, y, z = half_extents * 2  # Full dimensions
        ixx = mass * (y*y + z*z) / 12
        iyy = mass * (x*x + z*z) / 12
        izz = mass * (x*x + y*y) / 12

        assert abs(inertia.inertia[0, 0] - ixx) < 1e-10
        assert abs(inertia.inertia[1, 1] - iyy) < 1e-10
        assert abs(inertia.inertia[2, 2] - izz) < 1e-10

    def test_from_capsule_creates_valid_inertia(self):
        """Test capsule inertia computation."""
        mass = 2.0
        radius = 0.3
        half_length = 0.5
        inertia = Inertia.from_capsule(mass, radius, half_length)

        assert inertia.mass == mass
        np.testing.assert_array_equal(inertia.com, np.zeros(3))

        # Capsule inertia should be symmetric around z-axis
        assert abs(inertia.inertia[0, 0] - inertia.inertia[1, 1]) < 1e-10
        # All values should be positive
        assert inertia.inertia[0, 0] > 0
        assert inertia.inertia[2, 2] > 0

    def test_from_cylinder_creates_valid_inertia(self):
        """Test cylinder inertia computation."""
        mass = 1.5
        radius = 0.4
        half_length = 0.8
        inertia = Inertia.from_cylinder(mass, radius, half_length)

        assert inertia.mass == mass
        np.testing.assert_array_equal(inertia.com, np.zeros(3))

        # Cylinder inertia should be symmetric around z-axis
        assert abs(inertia.inertia[0, 0] - inertia.inertia[1, 1]) < 1e-10

    def test_unit_sphere_inertia(self, unit_mass_sphere_inertia):
        """Test unit sphere inertia fixture."""
        assert unit_mass_sphere_inertia.mass == 1.0
        expected_i = 2 * 1.0 * 1.0 * 1.0 / 5  # 0.4
        np.testing.assert_allclose(
            unit_mass_sphere_inertia.inertia,
            np.eye(3) * expected_i
        )


class TestGeom:
    """Tests for Geom class."""

    def test_sphere_geom_creation(self, sphere_geom):
        """Test sphere geometry creation."""
        assert sphere_geom.geom_type == GeomType.SPHERE
        assert sphere_geom.size[0] == 1.0
        assert sphere_geom.friction == 1.0
        assert sphere_geom.restitution == 0.0

    def test_box_geom_creation(self, box_geom):
        """Test box geometry creation."""
        assert box_geom.geom_type == GeomType.BOX
        np.testing.assert_array_equal(box_geom.size, np.array([0.5, 0.5, 0.5]))

    def test_plane_geom_creation(self, plane_geom):
        """Test plane geometry creation."""
        assert plane_geom.geom_type == GeomType.PLANE

    def test_sphere_aabb_at_origin(self, sphere_geom, identity_quat):
        """Test sphere AABB computation at origin."""
        body_pos = np.zeros(3)
        aabb_min, aabb_max = sphere_geom.get_aabb(body_pos, identity_quat)

        np.testing.assert_array_equal(aabb_min, np.array([-1.0, -1.0, -1.0]))
        np.testing.assert_array_equal(aabb_max, np.array([1.0, 1.0, 1.0]))

    def test_sphere_aabb_translated(self, sphere_geom, identity_quat):
        """Test sphere AABB computation when translated."""
        body_pos = np.array([5.0, 0.0, 3.0])
        aabb_min, aabb_max = sphere_geom.get_aabb(body_pos, identity_quat)

        np.testing.assert_array_equal(aabb_min, np.array([4.0, -1.0, 2.0]))
        np.testing.assert_array_equal(aabb_max, np.array([6.0, 1.0, 4.0]))

    def test_box_aabb_at_origin(self, box_geom, identity_quat):
        """Test box AABB computation at origin (no rotation)."""
        body_pos = np.zeros(3)
        aabb_min, aabb_max = box_geom.get_aabb(body_pos, identity_quat)

        np.testing.assert_allclose(aabb_min, np.array([-0.5, -0.5, -0.5]))
        np.testing.assert_allclose(aabb_max, np.array([0.5, 0.5, 0.5]))

    def test_capsule_aabb(self, capsule_geom, identity_quat):
        """Test capsule AABB computation."""
        body_pos = np.zeros(3)
        aabb_min, aabb_max = capsule_geom.get_aabb(body_pos, identity_quat)

        # Capsule: radius 0.5, half_length 1.0, aligned with Z
        expected_min = np.array([-0.5, -0.5, -1.5])
        expected_max = np.array([0.5, 0.5, 1.5])
        np.testing.assert_allclose(aabb_min, expected_min)
        np.testing.assert_allclose(aabb_max, expected_max)


class TestBody:
    """Tests for Body class."""

    def test_body_creation_with_defaults(self):
        """Test body creation with default values."""
        inertia = Inertia.from_sphere(1.0, 1.0)
        body = Body(name="test_body", inertia=inertia)

        assert body.name == "test_body"
        assert body.parent == -1
        assert body.geoms == []
        np.testing.assert_array_equal(body.pos, np.zeros(3))
        np.testing.assert_array_equal(body.quat, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_body_with_geom(self, sphere_body):
        """Test body with attached geometry."""
        assert len(sphere_body.geoms) == 1
        assert sphere_body.geoms[0].geom_type == GeomType.SPHERE

    def test_ground_body_has_zero_mass(self, ground_body):
        """Test ground body is static."""
        assert ground_body.inertia.mass == 0.0


class TestJoint:
    """Tests for Joint class."""

    def test_hinge_joint_creation(self):
        """Test hinge joint creation."""
        joint = Joint(
            joint_type=JointType.HINGE,
            parent_body=0,
            child_body=1,
            axis=np.array([0.0, 0.0, 1.0])
        )

        assert joint.joint_type == JointType.HINGE
        assert joint.parent_body == 0
        assert joint.child_body == 1
        np.testing.assert_array_equal(joint.axis, np.array([0.0, 0.0, 1.0]))

    def test_free_joint_dof(self):
        """Test free joint has correct DOF."""
        model = Model()
        body = Body(
            name="test",
            inertia=Inertia.from_sphere(1.0, 1.0)
        )
        model.add_body(body)

        joint = Joint(
            joint_type=JointType.FREE,
            parent_body=-1,
            child_body=0
        )
        model.add_joint(joint)

        assert joint.n_qpos == 7  # pos (3) + quat (4)
        assert joint.n_qvel == 6  # vel (3) + angular vel (3)

    def test_hinge_joint_dof(self):
        """Test hinge joint has 1 DOF."""
        model = Model()
        base = Body(name="base", inertia=Inertia.from_sphere(0.0, 0.0))
        model.add_body(base)
        arm = Body(name="arm", inertia=Inertia.from_sphere(1.0, 1.0), parent=0)
        model.add_body(arm)

        joint = Joint(
            joint_type=JointType.HINGE,
            parent_body=0,
            child_body=1
        )
        model.add_joint(joint)

        assert joint.n_qpos == 1
        assert joint.n_qvel == 1

    def test_slide_joint_dof(self):
        """Test slide joint has 1 DOF."""
        model = Model()
        base = Body(name="base", inertia=Inertia.from_sphere(0.0, 0.0))
        model.add_body(base)
        slider = Body(name="slider", inertia=Inertia.from_sphere(1.0, 1.0), parent=0)
        model.add_body(slider)

        joint = Joint(
            joint_type=JointType.SLIDE,
            parent_body=0,
            child_body=1
        )
        model.add_joint(joint)

        assert joint.n_qpos == 1
        assert joint.n_qvel == 1

    def test_ball_joint_dof(self):
        """Test ball joint has 3 rotational DOF."""
        model = Model()
        base = Body(name="base", inertia=Inertia.from_sphere(0.0, 0.0))
        model.add_body(base)
        ball = Body(name="ball", inertia=Inertia.from_sphere(1.0, 1.0), parent=0)
        model.add_body(ball)

        joint = Joint(
            joint_type=JointType.BALL,
            parent_body=0,
            child_body=1
        )
        model.add_joint(joint)

        assert joint.n_qpos == 4  # quaternion
        assert joint.n_qvel == 3  # angular velocity

    def test_fixed_joint_dof(self):
        """Test fixed joint has 0 DOF."""
        model = Model()
        base = Body(name="base", inertia=Inertia.from_sphere(0.0, 0.0))
        model.add_body(base)
        fixed = Body(name="fixed", inertia=Inertia.from_sphere(1.0, 1.0), parent=0)
        model.add_body(fixed)

        joint = Joint(
            joint_type=JointType.FIXED,
            parent_body=0,
            child_body=1
        )
        model.add_joint(joint)

        assert joint.n_qpos == 0
        assert joint.n_qvel == 0


class TestActuator:
    """Tests for Actuator class."""

    def test_actuator_creation(self):
        """Test actuator creation."""
        actuator = Actuator(
            joint_idx=0,
            gear=10.0,
            ctrl_range=(-1.0, 1.0)
        )

        assert actuator.joint_idx == 0
        assert actuator.gear == 10.0
        assert actuator.ctrl_range == (-1.0, 1.0)

    def test_actuator_defaults(self):
        """Test actuator default values."""
        actuator = Actuator(joint_idx=0)

        assert actuator.gear == 1.0
        assert actuator.ctrl_range == (-1.0, 1.0)
        assert actuator.damping == 0.0


class TestModel:
    """Tests for Model class."""

    def test_empty_model_creation(self, empty_model):
        """Test empty model creation."""
        assert empty_model.nbody == 0
        assert empty_model.nq == 0
        assert empty_model.nv == 0
        assert empty_model.nu == 0

    def test_add_body_increments_nbody(self, empty_model):
        """Test adding body increments nbody."""
        body = Body(name="test", inertia=Inertia.from_sphere(1.0, 1.0))
        idx = empty_model.add_body(body)

        assert idx == 0
        assert empty_model.nbody == 1

    def test_add_joint_updates_dof(self, empty_model):
        """Test adding joint updates DOF counts."""
        body = Body(name="test", inertia=Inertia.from_sphere(1.0, 1.0))
        empty_model.add_body(body)

        joint = Joint(
            joint_type=JointType.FREE,
            parent_body=-1,
            child_body=0
        )
        empty_model.add_joint(joint)

        assert empty_model.nq == 7
        assert empty_model.nv == 6

    def test_add_actuator_updates_nu(self, pendulum_model):
        """Test adding actuator updates control dimension."""
        assert pendulum_model.nu == 1

    def test_model_gravity_default(self, empty_model):
        """Test default gravity vector."""
        np.testing.assert_array_equal(
            empty_model.gravity,
            np.array([0.0, 0.0, -9.81])
        )

    def test_single_free_body_model(self, single_free_body_model):
        """Test single free body model structure."""
        assert single_free_body_model.nbody == 1
        assert single_free_body_model.nq == 7
        assert single_free_body_model.nv == 6
        assert len(single_free_body_model.joints) == 1


class TestState:
    """Tests for State class."""

    def test_state_creation(self, single_free_body_model):
        """Test state creation from model."""
        state = State.create(single_free_body_model)

        assert state.time == 0.0
        assert state.qpos.shape == (7,)
        assert state.qvel.shape == (6,)
        assert state.qacc.shape == (6,)
        assert state.ctrl.shape == (0,)  # No actuators

    def test_state_initial_body_positions(self, single_free_body_model):
        """Test state initializes body positions."""
        state = State.create(single_free_body_model)

        assert state.xpos.shape == (1, 3)
        assert state.xquat.shape == (1, 4)
        # Identity quaternion
        np.testing.assert_array_equal(state.xquat[0], np.array([1, 0, 0, 0]))

    def test_state_contacts_empty(self, initial_state):
        """Test state contacts initially empty."""
        assert initial_state.contacts == []


class TestContact:
    """Tests for Contact class."""

    def test_contact_creation(self, sample_contact):
        """Test contact creation."""
        np.testing.assert_array_equal(
            sample_contact.pos,
            np.array([0.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(
            sample_contact.normal,
            np.array([0.0, 0.0, 1.0])
        )
        assert sample_contact.penetration == 0.1
        assert sample_contact.friction == 1.0
        assert sample_contact.restitution == 0.0


class TestQuaternionUtilities:
    """Tests for quaternion utility functions."""

    def test_quat_mul_identity(self, identity_quat):
        """Test quaternion multiplication with identity."""
        q = np.array([0.707, 0.707, 0.0, 0.0])  # 90 deg around X
        result = quat_mul(identity_quat, q)
        np.testing.assert_allclose(result, q, atol=1e-6)

    def test_quat_mul_inverse(self):
        """Test quaternion times its conjugate gives identity (up to normalization)."""
        # Use a normalized quaternion for this test
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])
        q_conj = quat_conj(q)
        result = quat_mul(q, q_conj)
        # Result should be [1, 0, 0, 0] (identity)
        np.testing.assert_allclose(result, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    def test_quat_rotate_identity(self, identity_quat):
        """Test rotation with identity quaternion."""
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate(identity_quat, v)
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_quat_rotate_90_deg_z(self):
        """Test 90 degree rotation around Z axis."""
        # 90 degrees around Z: [cos(45), 0, 0, sin(45)]
        q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        v = np.array([1.0, 0.0, 0.0])

        result = quat_rotate(q, v)
        expected = np.array([0.0, 1.0, 0.0])  # X -> Y
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_quat_rotate_180_deg(self):
        """Test 180 degree rotation around X axis."""
        # 180 degrees around X: [0, 1, 0, 0]
        q = np.array([0.0, 1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])

        result = quat_rotate(q, v)
        expected = np.array([0.0, -1.0, 0.0])  # Y -> -Y
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_quat_conj(self):
        """Test quaternion conjugate."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = quat_conj(q)
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        np.testing.assert_array_equal(result, expected)

    def test_axis_angle_to_quat_zero_angle(self):
        """Test axis-angle conversion with zero angle."""
        axis = np.array([0.0, 0.0, 1.0])
        q = axis_angle_to_quat(axis, 0.0)
        np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-10)

    def test_axis_angle_to_quat_90_deg(self):
        """Test axis-angle conversion with 90 degrees."""
        axis = np.array([0.0, 0.0, 1.0])
        q = axis_angle_to_quat(axis, np.pi / 2)

        # w = cos(45), z = sin(45)
        expected = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_quat_to_mat_identity(self, identity_quat):
        """Test quaternion to rotation matrix for identity."""
        R = quat_to_mat(identity_quat)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_quat_to_mat_rotation(self):
        """Test quaternion to rotation matrix consistency."""
        # 90 degrees around Z
        q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        R = quat_to_mat(q)

        # Apply rotation via matrix and quaternion, should match
        v = np.array([1.0, 0.0, 0.0])
        result_mat = R @ v
        result_quat = quat_rotate(q, v)

        np.testing.assert_allclose(result_mat, result_quat, atol=1e-10)


class TestGeomTypes:
    """Tests for GeomType enum."""

    def test_geom_types_exist(self):
        """Test all geometry types exist."""
        assert GeomType.SPHERE.value == "sphere"
        assert GeomType.BOX.value == "box"
        assert GeomType.CAPSULE.value == "capsule"
        assert GeomType.CYLINDER.value == "cylinder"
        assert GeomType.PLANE.value == "plane"
        assert GeomType.MESH.value == "mesh"


class TestJointTypes:
    """Tests for JointType enum."""

    def test_joint_types_exist(self):
        """Test all joint types exist."""
        assert JointType.FREE.value == "free"
        assert JointType.BALL.value == "ball"
        assert JointType.HINGE.value == "hinge"
        assert JointType.SLIDE.value == "slide"
        assert JointType.FIXED.value == "fixed"


class TestModelCompilation:
    """Tests for model compilation."""

    def test_model_compile_does_not_error(self, pendulum_model):
        """Test model compile method runs without error."""
        pendulum_model.compile()  # Should not raise

    def test_cart_pole_model_structure(self, cart_pole_model):
        """Test cart-pole model has correct structure."""
        assert cart_pole_model.nbody == 3  # ground, cart, pole
        assert len(cart_pole_model.joints) == 2  # slide, hinge
        assert cart_pole_model.nu == 1  # one actuator
        assert cart_pole_model.nq == 2  # slide + hinge positions
        assert cart_pole_model.nv == 2  # slide + hinge velocities
