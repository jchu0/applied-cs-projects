"""Pytest fixtures for PhysicsRL test suite."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np

from physicsrl import (
    Model, Body, Joint, Geom, Actuator, Inertia, State,
    GeomType, JointType, Contact
)
from physicsrl.dynamics import ForwardDynamics
from physicsrl.collision import BroadPhase, NarrowPhase, CollisionSystem
from physicsrl.solver import ConstraintSolver
from physicsrl.integration import Integrator


# =============================================================================
# Basic Fixture Factories
# =============================================================================

@pytest.fixture
def identity_quat():
    """Identity quaternion [w, x, y, z]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def zero_vector():
    """Zero 3D vector."""
    return np.zeros(3)


# =============================================================================
# Inertia Fixtures
# =============================================================================

@pytest.fixture
def unit_mass_sphere_inertia():
    """Inertia for a unit sphere (mass=1, radius=1)."""
    return Inertia.from_sphere(mass=1.0, radius=1.0)


@pytest.fixture
def unit_mass_box_inertia():
    """Inertia for a unit box (mass=1, half_extents=[0.5, 0.5, 0.5])."""
    return Inertia.from_box(mass=1.0, half_extents=np.array([0.5, 0.5, 0.5]))


@pytest.fixture
def unit_mass_capsule_inertia():
    """Inertia for a unit capsule (mass=1, radius=0.5, half_length=1.0)."""
    return Inertia.from_capsule(mass=1.0, radius=0.5, half_length=1.0)


@pytest.fixture
def unit_mass_cylinder_inertia():
    """Inertia for a unit cylinder (mass=1, radius=0.5, half_length=1.0)."""
    return Inertia.from_cylinder(mass=1.0, radius=0.5, half_length=1.0)


# =============================================================================
# Geometry Fixtures
# =============================================================================

@pytest.fixture
def sphere_geom():
    """Sphere geometry at origin with radius 1."""
    return Geom(
        geom_type=GeomType.SPHERE,
        size=np.array([1.0]),
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        friction=1.0,
        restitution=0.0
    )


@pytest.fixture
def box_geom():
    """Box geometry at origin with half_extents [0.5, 0.5, 0.5]."""
    return Geom(
        geom_type=GeomType.BOX,
        size=np.array([0.5, 0.5, 0.5]),
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        friction=1.0,
        restitution=0.0
    )


@pytest.fixture
def plane_geom():
    """Plane geometry at origin (XY plane, Z up)."""
    return Geom(
        geom_type=GeomType.PLANE,
        size=np.array([10.0, 10.0, 0.1]),
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        friction=1.0,
        restitution=0.0
    )


@pytest.fixture
def capsule_geom():
    """Capsule geometry at origin with radius 0.5 and half_length 1.0."""
    return Geom(
        geom_type=GeomType.CAPSULE,
        size=np.array([0.5, 1.0]),
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        friction=1.0,
        restitution=0.0
    )


@pytest.fixture
def cylinder_geom():
    """Cylinder geometry at origin with radius 0.5 and half_length 1.0."""
    return Geom(
        geom_type=GeomType.CYLINDER,
        size=np.array([0.5, 1.0]),
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        friction=1.0,
        restitution=0.0
    )


# =============================================================================
# Body Fixtures
# =============================================================================

@pytest.fixture
def ground_body(plane_geom):
    """Ground plane body with zero mass (static)."""
    return Body(
        name="ground",
        inertia=Inertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        parent=-1,
        geoms=[plane_geom],
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )


@pytest.fixture
def sphere_body(sphere_geom, unit_mass_sphere_inertia):
    """Sphere body at height 2."""
    return Body(
        name="sphere",
        inertia=unit_mass_sphere_inertia,
        parent=-1,
        geoms=[sphere_geom],
        pos=np.array([0.0, 0.0, 2.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )


@pytest.fixture
def box_body(box_geom, unit_mass_box_inertia):
    """Box body at height 2."""
    return Body(
        name="box",
        inertia=unit_mass_box_inertia,
        parent=-1,
        geoms=[box_geom],
        pos=np.array([0.0, 0.0, 2.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def empty_model():
    """Empty model with default settings."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "euler"
    return model


@pytest.fixture
def single_free_body_model(sphere_body):
    """Model with a single free-floating sphere."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "euler"

    # Add sphere body
    model.add_body(sphere_body)

    # Add free joint
    joint = Joint(
        joint_type=JointType.FREE,
        parent_body=-1,
        child_body=0
    )
    model.add_joint(joint)

    return model


@pytest.fixture
def falling_sphere_model(ground_body, sphere_body):
    """Model with ground plane and falling sphere."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "semi_implicit"

    # Add ground
    model.add_body(ground_body)

    # Add sphere
    sphere_body.pos = np.array([0.0, 0.0, 2.0])
    model.add_body(sphere_body)

    # Add free joint for sphere
    joint = Joint(
        joint_type=JointType.FREE,
        parent_body=-1,
        child_body=1  # sphere is second body
    )
    model.add_joint(joint)

    return model


@pytest.fixture
def pendulum_model():
    """Simple pendulum model with hinge joint."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "semi_implicit"

    # Base body (fixed)
    base = Body(
        name="base",
        inertia=Inertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        parent=-1,
        geoms=[],
        pos=np.array([0.0, 0.0, 2.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(base)

    # Pendulum body
    pendulum = Body(
        name="pendulum",
        inertia=Inertia.from_box(1.0, np.array([0.1, 0.1, 0.5])),
        parent=0,
        geoms=[Geom(GeomType.BOX, np.array([0.1, 0.1, 0.5]))],
        pos=np.array([0.0, 0.0, 1.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(pendulum)

    # Hinge joint
    joint = Joint(
        joint_type=JointType.HINGE,
        parent_body=0,
        child_body=1,
        pos=np.array([0.0, 0.0, 0.0]),
        axis=np.array([1.0, 0.0, 0.0]),  # Rotate around X
        damping=0.1
    )
    model.add_joint(joint)

    # Actuator on hinge
    actuator = Actuator(
        joint_idx=0,
        gear=1.0,
        ctrl_range=(-1.0, 1.0)
    )
    model.add_actuator(actuator)

    return model


@pytest.fixture
def two_sphere_model():
    """Model with two spheres for collision testing."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "euler"

    # Sphere 1
    sphere1 = Body(
        name="sphere1",
        inertia=Inertia.from_sphere(1.0, 1.0),
        parent=-1,
        geoms=[Geom(GeomType.SPHERE, np.array([1.0]))],
        pos=np.array([-1.5, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(sphere1)

    joint1 = Joint(
        joint_type=JointType.FREE,
        parent_body=-1,
        child_body=0
    )
    model.add_joint(joint1)

    # Sphere 2
    sphere2 = Body(
        name="sphere2",
        inertia=Inertia.from_sphere(1.0, 1.0),
        parent=-1,
        geoms=[Geom(GeomType.SPHERE, np.array([1.0]))],
        pos=np.array([1.5, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(sphere2)

    joint2 = Joint(
        joint_type=JointType.FREE,
        parent_body=-1,
        child_body=1
    )
    model.add_joint(joint2)

    return model


@pytest.fixture
def cart_pole_model():
    """Cart-pole model with slide and hinge joints."""
    model = Model()
    model.timestep = 0.002
    model.integrator = "semi_implicit"

    # Ground body (fixed)
    ground = Body(
        name="ground",
        inertia=Inertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        parent=-1,
        geoms=[],
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(ground)

    # Cart body
    cart = Body(
        name="cart",
        inertia=Inertia.from_box(1.0, np.array([0.5, 0.25, 0.1])),
        parent=0,
        geoms=[Geom(GeomType.BOX, np.array([0.5, 0.25, 0.1]))],
        pos=np.array([0.0, 0.0, 0.1]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(cart)

    # Slide joint for cart
    slide_joint = Joint(
        joint_type=JointType.SLIDE,
        parent_body=0,
        child_body=1,
        pos=np.zeros(3),
        axis=np.array([1.0, 0.0, 0.0]),  # Slide along X
        limit_lower=-2.4,
        limit_upper=2.4
    )
    model.add_joint(slide_joint)

    # Pole body
    pole = Body(
        name="pole",
        inertia=Inertia.from_capsule(0.1, 0.05, 0.5),
        parent=1,
        geoms=[Geom(GeomType.CAPSULE, np.array([0.05, 0.5]))],
        pos=np.array([0.0, 0.0, 0.6]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(pole)

    # Hinge joint for pole
    hinge_joint = Joint(
        joint_type=JointType.HINGE,
        parent_body=1,
        child_body=2,
        pos=np.array([0.0, 0.0, 0.1]),
        axis=np.array([0.0, 1.0, 0.0]),  # Rotate around Y
        damping=0.01
    )
    model.add_joint(hinge_joint)

    # Actuator on cart
    actuator = Actuator(
        joint_idx=0,
        gear=10.0,
        ctrl_range=(-1.0, 1.0)
    )
    model.add_actuator(actuator)

    return model


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def initial_state(single_free_body_model):
    """Initial state for single free body model."""
    return State.create(single_free_body_model)


# =============================================================================
# Dynamics Fixtures
# =============================================================================

@pytest.fixture
def forward_dynamics(single_free_body_model):
    """ForwardDynamics instance for single free body model."""
    return ForwardDynamics(single_free_body_model)


# =============================================================================
# Collision Fixtures
# =============================================================================

@pytest.fixture
def broad_phase():
    """BroadPhase instance."""
    return BroadPhase()


@pytest.fixture
def narrow_phase():
    """NarrowPhase instance."""
    return NarrowPhase()


@pytest.fixture
def collision_system(falling_sphere_model):
    """CollisionSystem instance for falling sphere model."""
    return CollisionSystem(falling_sphere_model)


# =============================================================================
# Solver Fixtures
# =============================================================================

@pytest.fixture
def constraint_solver(falling_sphere_model):
    """ConstraintSolver instance."""
    return ConstraintSolver(falling_sphere_model)


# =============================================================================
# Integrator Fixtures
# =============================================================================

@pytest.fixture
def simple_hinge_model():
    """Simple hinge joint model for 1-DOF integration testing.

    (Free-joint stepping, where qpos has 7 components and qvel has 6, is now
    integrated correctly and is exercised directly in
    tests/test_integration.py::TestFreeBodyIntegration.)
    """
    model = Model()
    model.timestep = 0.002
    model.integrator = "euler"

    # Base body (fixed)
    base = Body(
        name="base",
        inertia=Inertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        parent=-1,
        geoms=[],
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(base)

    # Arm body
    arm = Body(
        name="arm",
        inertia=Inertia.from_box(1.0, np.array([0.1, 0.1, 0.5])),
        parent=0,
        geoms=[Geom(GeomType.BOX, np.array([0.1, 0.1, 0.5]))],
        pos=np.array([0.0, 0.0, 0.5]),
        quat=np.array([1.0, 0.0, 0.0, 0.0])
    )
    model.add_body(arm)

    # Hinge joint
    joint = Joint(
        joint_type=JointType.HINGE,
        parent_body=0,
        child_body=1,
        pos=np.zeros(3),
        axis=np.array([1.0, 0.0, 0.0]),
        damping=0.0
    )
    model.add_joint(joint)

    return model


@pytest.fixture
def euler_integrator(simple_hinge_model):
    """Euler integrator for simple hinge model."""
    simple_hinge_model.integrator = "euler"
    return Integrator(simple_hinge_model)


@pytest.fixture
def rk4_integrator(simple_hinge_model):
    """RK4 integrator for simple hinge model."""
    simple_hinge_model.integrator = "rk4"
    return Integrator(simple_hinge_model)


@pytest.fixture
def semi_implicit_integrator(simple_hinge_model):
    """Semi-implicit integrator for simple hinge model."""
    simple_hinge_model.integrator = "semi_implicit"
    return Integrator(simple_hinge_model)


# =============================================================================
# Contact Fixtures
# =============================================================================

@pytest.fixture
def sample_contact():
    """Sample contact for testing."""
    return Contact(
        pos=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        penetration=0.1,
        geom1_idx=0,
        geom2_idx=1,
        body1_idx=0,
        body2_idx=1,
        friction=1.0,
        restitution=0.0
    )


# =============================================================================
# Utility Functions for Tests
# =============================================================================

@pytest.fixture
def assert_quaternion_normalized():
    """Fixture that returns a function to check quaternion normalization."""
    def _check(q, tol=1e-6):
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < tol, f"Quaternion not normalized: norm={norm}"
    return _check


@pytest.fixture
def assert_arrays_close():
    """Fixture that returns a function to check array closeness."""
    def _check(a, b, rtol=1e-5, atol=1e-8):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return _check
