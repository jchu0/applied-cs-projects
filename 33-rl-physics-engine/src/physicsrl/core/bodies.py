"""Core rigid body data structures."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class GeomType(Enum):
    """Collision geometry types."""
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    PLANE = "plane"
    MESH = "mesh"


class JointType(Enum):
    """Joint constraint types."""
    FREE = "free"       # 6 DOF
    BALL = "ball"       # 3 DOF (rotation)
    HINGE = "hinge"     # 1 DOF (rotation)
    SLIDE = "slide"     # 1 DOF (translation)
    FIXED = "fixed"     # 0 DOF


@dataclass
class Inertia:
    """Rigid body inertia properties."""
    mass: float
    com: np.ndarray  # Center of mass in body frame (3,)
    inertia: np.ndarray  # Inertia tensor (3, 3)

    @staticmethod
    def from_box(mass: float, half_extents: np.ndarray) -> 'Inertia':
        """Create inertia for box geometry."""
        x, y, z = half_extents * 2  # Full dimensions
        ixx = mass * (y*y + z*z) / 12
        iyy = mass * (x*x + z*z) / 12
        izz = mass * (x*x + y*y) / 12
        return Inertia(
            mass=mass,
            com=np.zeros(3),
            inertia=np.diag([ixx, iyy, izz])
        )

    @staticmethod
    def from_sphere(mass: float, radius: float) -> 'Inertia':
        """Create inertia for sphere geometry."""
        i = 2 * mass * radius * radius / 5
        return Inertia(
            mass=mass,
            com=np.zeros(3),
            inertia=np.eye(3) * i
        )

    @staticmethod
    def from_capsule(mass: float, radius: float, half_length: float) -> 'Inertia':
        """Create inertia for capsule geometry."""
        r2 = radius * radius
        h = half_length * 2

        # Cylinder part
        m_cyl = mass * h / (h + 4*radius/3)
        i_cyl_xx = m_cyl * (3*r2 + h*h) / 12
        i_cyl_zz = m_cyl * r2 / 2

        # Hemisphere caps
        m_cap = (mass - m_cyl) / 2
        i_cap_xx = m_cap * (2*r2/5 + h*h/4 + 3*h*radius/8)
        i_cap_zz = m_cap * 2 * r2 / 5

        ixx = i_cyl_xx + 2*i_cap_xx
        izz = i_cyl_zz + 2*i_cap_zz

        return Inertia(
            mass=mass,
            com=np.zeros(3),
            inertia=np.diag([ixx, ixx, izz])
        )

    @staticmethod
    def from_cylinder(mass: float, radius: float, half_length: float) -> 'Inertia':
        """Create inertia for cylinder geometry."""
        r2 = radius * radius
        h2 = (2 * half_length) ** 2
        ixx = mass * (3*r2 + h2) / 12
        izz = mass * r2 / 2
        return Inertia(
            mass=mass,
            com=np.zeros(3),
            inertia=np.diag([ixx, ixx, izz])
        )


@dataclass
class Contact:
    """Contact point between two geometries."""
    pos: np.ndarray      # Contact position in world frame
    normal: np.ndarray   # Contact normal (from geom1 to geom2)
    penetration: float   # Penetration depth (positive = overlap)
    geom1_idx: int
    geom2_idx: int
    body1_idx: int
    body2_idx: int

    # Contact dynamics
    friction: float = 1.0
    restitution: float = 0.0


@dataclass
class Geom:
    """Collision geometry attached to body."""
    geom_type: GeomType
    size: np.ndarray  # Geometry-specific size parameters
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Local position
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Local orientation

    # Contact properties
    friction: float = 1.0
    restitution: float = 0.0

    # Optional mesh data
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    def get_aabb(self, body_pos: np.ndarray, body_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box in world frame."""
        # Transform to world frame
        world_pos = body_pos + quat_rotate(body_quat, self.pos)

        # Compute AABB based on geometry type
        if self.geom_type == GeomType.SPHERE:
            radius = self.size[0]
            return world_pos - radius, world_pos + radius

        elif self.geom_type == GeomType.BOX:
            # Rotate box corners and find extents
            half = self.size
            corners = np.array([
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
            ], dtype=np.float64) * half

            # Apply rotation
            world_quat = quat_mul(body_quat, self.quat)
            rotated = np.array([quat_rotate(world_quat, c) for c in corners])

            min_pt = world_pos + rotated.min(axis=0)
            max_pt = world_pos + rotated.max(axis=0)
            return min_pt, max_pt

        elif self.geom_type == GeomType.CAPSULE:
            radius = self.size[0]
            half_length = self.size[1]

            # Capsule endpoints
            endpoints = np.array([[0, 0, -half_length], [0, 0, half_length]])
            world_quat = quat_mul(body_quat, self.quat)
            rotated = np.array([quat_rotate(world_quat, e) for e in endpoints])

            min_pt = world_pos + rotated.min(axis=0) - radius
            max_pt = world_pos + rotated.max(axis=0) + radius
            return min_pt, max_pt

        elif self.geom_type == GeomType.CYLINDER:
            radius = self.size[0]
            half_length = self.size[1]

            # Similar to capsule
            endpoints = np.array([[0, 0, -half_length], [0, 0, half_length]])
            world_quat = quat_mul(body_quat, self.quat)
            rotated = np.array([quat_rotate(world_quat, e) for e in endpoints])

            min_pt = world_pos + rotated.min(axis=0) - radius
            max_pt = world_pos + rotated.max(axis=0) + radius
            return min_pt, max_pt

        elif self.geom_type == GeomType.PLANE:
            # Infinite plane - use large bounds
            large = 1000.0
            return world_pos - large, world_pos + large

        else:
            # Default: use size as half-extents
            return world_pos - self.size, world_pos + self.size


@dataclass
class Joint:
    """Joint connecting two bodies."""
    joint_type: JointType
    parent_body: int  # -1 for world
    child_body: int

    # Joint frame in parent body
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    # Joint limits
    limit_lower: float = -np.inf
    limit_upper: float = np.inf
    damping: float = 0.0
    stiffness: float = 0.0

    # Joint state indices
    qpos_idx: int = 0  # Index in qpos
    qvel_idx: int = 0  # Index in qvel
    n_qpos: int = 1    # DOF in position
    n_qvel: int = 1    # DOF in velocity


@dataclass
class Actuator:
    """Actuator that applies forces/torques."""
    joint_idx: int
    gear: float = 1.0
    ctrl_range: Tuple[float, float] = (-1.0, 1.0)

    # Dynamics parameters
    forcerange: Tuple[float, float] = (-np.inf, np.inf)
    damping: float = 0.0


@dataclass
class Body:
    """Rigid body in the simulation."""
    name: str
    inertia: Inertia
    parent: int = -1  # Parent body index (-1 for root)

    # Geometry
    geoms: List[Geom] = field(default_factory=list)

    # Initial state
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


class Model:
    """Complete simulation model."""

    def __init__(self):
        self.bodies: List[Body] = []
        self.joints: List[Joint] = []
        self.actuators: List[Actuator] = []

        # Computed dimensions
        self.nq = 0  # Position DOF
        self.nv = 0  # Velocity DOF
        self.nu = 0  # Control dimension
        self.nbody = 0

        # Gravity
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Simulation parameters
        self.timestep = 0.002
        self.integrator = "euler"  # "euler", "rk4", "semi_implicit"

    def add_body(self, body: Body) -> int:
        """Add a body to the model."""
        idx = len(self.bodies)
        self.bodies.append(body)
        self.nbody += 1
        return idx

    def add_joint(self, joint: Joint) -> int:
        """Add a joint to the model."""
        # Compute DOF based on joint type
        if joint.joint_type == JointType.FREE:
            joint.n_qpos = 7  # pos + quaternion
            joint.n_qvel = 6  # vel + angular vel
        elif joint.joint_type == JointType.BALL:
            joint.n_qpos = 4  # quaternion
            joint.n_qvel = 3  # angular vel
        elif joint.joint_type in {JointType.HINGE, JointType.SLIDE}:
            joint.n_qpos = 1
            joint.n_qvel = 1
        else:  # FIXED
            joint.n_qpos = 0
            joint.n_qvel = 0

        joint.qpos_idx = self.nq
        joint.qvel_idx = self.nv
        self.nq += joint.n_qpos
        self.nv += joint.n_qvel

        idx = len(self.joints)
        self.joints.append(joint)
        return idx

    def add_actuator(self, actuator: Actuator) -> int:
        """Add an actuator to the model."""
        idx = len(self.actuators)
        self.actuators.append(actuator)
        self.nu += 1
        return idx

    def compile(self):
        """Compile model for simulation."""
        # Build kinematic tree
        # Compute connectivity
        # Pre-compute transforms
        pass


@dataclass
class State:
    """Simulation state."""
    time: float = 0.0
    qpos: np.ndarray = None  # Generalized positions
    qvel: np.ndarray = None  # Generalized velocities
    qacc: np.ndarray = None  # Generalized accelerations
    ctrl: np.ndarray = None  # Control inputs

    # Computed quantities
    xpos: np.ndarray = None   # Body positions
    xquat: np.ndarray = None  # Body orientations
    xvel: np.ndarray = None   # Body velocities

    # Contact state
    contacts: List[Contact] = field(default_factory=list)

    @staticmethod
    def create(model: Model) -> 'State':
        """Create initial state for model."""
        state = State()
        state.qpos = np.zeros(model.nq)
        state.qvel = np.zeros(model.nv)
        state.qacc = np.zeros(model.nv)
        state.ctrl = np.zeros(model.nu)

        state.xpos = np.zeros((model.nbody, 3))
        state.xquat = np.zeros((model.nbody, 4))
        state.xquat[:, 0] = 1  # Identity quaternion
        state.xvel = np.zeros((model.nbody, 6))

        # Initialize from model
        for i, body in enumerate(model.bodies):
            state.xpos[i] = body.pos
            state.xquat[i] = body.quat

        return state


# Quaternion utilities
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quat_mul(quat_mul(q, qv), q_conj)
    return result[1:]


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to quaternion."""
    half = angle / 2
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
