# Deep Reinforcement Learning Physics Engine (MuJoCo-lite) - Technical Blueprint

> **Concepts covered:** §03 ml-engineering — `01-ml-fundamentals`, `02-deep-learning`

## Executive Summary

This project implements a high-performance physics engine optimized for reinforcement learning applications. It features rigid body dynamics with contact resolution, batch simulation for parallel training, GPU acceleration, and a clean observation/action API suitable for RL environments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL Physics Engine Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          User API Layer                               │   │
│  │   Environment   Observation    Action     Reward     Reset/Step      │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     World Model & Scene Graph                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │    Bodies    │  │    Joints    │  │   Actuators  │               │   │
│  │  │  (Position,  │  │  (Hinge,     │  │   (Motors,   │               │   │
│  │  │   Inertia)   │  │   Ball)      │  │    Forces)   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Physics Core Engine                             │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │   │
│  │  │   Forward     │  │  Constraint   │  │   Contact     │             │   │
│  │  │   Dynamics    │──│   Solver      │──│   Resolution  │             │   │
│  │  │  (RNE/ABA)    │  │  (Gauss-Seid) │  │   (LCP)       │             │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘             │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Collision Detection                               │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │   │
│  │  │  Broad Phase  │  │  Narrow Phase │  │   Contact     │             │   │
│  │  │  (AABB Tree)  │──│  (GJK/EPA)    │──│   Manifold    │             │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘             │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  Numerical Integration & GPU                          │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │   │
│  │  │   Integrator  │  │   Batch Sim   │  │     GPU       │             │   │
│  │  │  (RK4/Euler)  │  │  (Parallel)   │  │  Accelerate   │             │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Rigid Body Model

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import quaternion  # numpy-quaternion

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
        # Cylinder + hemisphere caps
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

@dataclass
class Geom:
    """Collision geometry attached to body."""
    geom_type: GeomType
    size: np.ndarray  # Geometry-specific size parameters
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Local position
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # Local orientation

    # Contact properties
    friction: float = 1.0
    restitution: float = 0.0

    # Optional mesh data
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    def get_aabb(self, body_pos: np.ndarray, body_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box in world frame."""
        # Transform to world frame
        world_pos = body_pos + quaternion.rotate_vectors(
            quaternion.from_float_array(body_quat),
            self.pos
        )

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
            ]) * half

            # Apply rotation
            world_quat = quaternion.from_float_array(body_quat) * \
                         quaternion.from_float_array(self.quat)
            rotated = quaternion.rotate_vectors(world_quat, corners)

            min_pt = world_pos + rotated.min(axis=0)
            max_pt = world_pos + rotated.max(axis=0)
            return min_pt, max_pt

        elif self.geom_type == GeomType.CAPSULE:
            radius = self.size[0]
            half_length = self.size[1]

            # Capsule endpoints
            endpoints = np.array([[0, 0, -half_length], [0, 0, half_length]])
            world_quat = quaternion.from_float_array(body_quat) * \
                         quaternion.from_float_array(self.quat)
            rotated = quaternion.rotate_vectors(world_quat, endpoints)

            min_pt = world_pos + rotated.min(axis=0) - radius
            max_pt = world_pos + rotated.max(axis=0) + radius
            return min_pt, max_pt

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
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))

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
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))

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
        self.gravity = np.array([0, 0, -9.81])

        # Simulation parameters
        self.timestep = 0.002
        self.integrator = "euler"  # "euler", "rk4", "implicit"

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
    contacts: List['Contact'] = field(default_factory=list)

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
```

### 2. Forward Dynamics Engine

```python
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
                a_parent = self._get_body_acc(body.parent)

            # Find joint for this body
            joint = None
            for j in self.model.joints:
                if j.child_body == i:
                    joint = j
                    break

            if joint is None:
                continue

            # Compute joint acceleration
            S = self._joint_motion_subspace(joint)
            c = self._velocity_product(state, i)  # Velocity-dependent term

            # Articulated body equation
            # tau = H * qacc + c
            # where H is the inertia in the joint DOF

            H = S.T @ self.spatial_inertia[i] @ S
            h = S.T @ (self.spatial_inertia[i] @ a_parent + self.bias_force[i])

            joint_tau = tau[joint.qvel_idx:joint.qvel_idx + joint.n_qvel]
            joint_qacc = np.linalg.solve(H, joint_tau - h)

            qacc[joint.qvel_idx:joint.qvel_idx + joint.n_qvel] = joint_qacc

        return qacc

    def _forward_kinematics(self, state: State):
        """Compute body positions and velocities from joint state."""
        model = self.model

        for i in range(model.nbody):
            body = model.bodies[i]

            if body.parent == -1:
                # Root body - position from qpos
                state.xpos[i] = state.qpos[:3]
                state.xquat[i] = state.qpos[3:7]
            else:
                # Find joint connecting to parent
                for joint in model.joints:
                    if joint.child_body == i:
                        # Compute transform based on joint type and position
                        parent_pos = state.xpos[body.parent]
                        parent_quat = state.xquat[body.parent]

                        if joint.joint_type == JointType.HINGE:
                            angle = state.qpos[joint.qpos_idx]
                            q = self._axis_angle_to_quat(joint.axis, angle)
                            state.xquat[i] = self._quat_mul(parent_quat, q)
                            state.xpos[i] = parent_pos + self._quat_rotate(
                                parent_quat, joint.pos
                            )

                        elif joint.joint_type == JointType.SLIDE:
                            offset = joint.axis * state.qpos[joint.qpos_idx]
                            state.xquat[i] = parent_quat.copy()
                            state.xpos[i] = parent_pos + self._quat_rotate(
                                parent_quat, joint.pos + offset
                            )

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
        # [I - m*c_cross*c_cross,  m*c_cross]
        # [m*c_cross^T,            m*I_3x3  ]
        c_cross = self._skew(c)

        spatial = np.zeros((6, 6))
        spatial[:3, :3] = I - m * c_cross @ c_cross
        spatial[:3, 3:] = m * c_cross
        spatial[3:, :3] = m * c_cross.T
        spatial[3:, 3:] = m * np.eye(3)

        return spatial

    def _joint_motion_subspace(self, joint: Joint) -> np.ndarray:
        """Get motion subspace matrix for joint."""
        if joint.joint_type == JointType.HINGE:
            # Rotation about axis
            S = np.zeros((6, 1))
            S[:3, 0] = joint.axis
            return S

        elif joint.joint_type == JointType.SLIDE:
            # Translation along axis
            S = np.zeros((6, 1))
            S[3:, 0] = joint.axis
            return S

        elif joint.joint_type == JointType.BALL:
            # 3 DOF rotation
            return np.vstack([np.eye(3), np.zeros((3, 3))])

        elif joint.joint_type == JointType.FREE:
            # Full 6 DOF
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
    def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion."""
        half = angle / 2
        s = np.sin(half)
        return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        qv = np.array([0, v[0], v[1], v[2]])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        result = ForwardDynamics._quat_mul(
            ForwardDynamics._quat_mul(q, qv), q_conj
        )
        return result[1:]

    def _velocity_product(self, state: State, body_idx: int) -> np.ndarray:
        """Compute velocity-dependent acceleration term."""
        # Coriolis and centrifugal terms
        return np.zeros(6)  # Simplified

    def _transform_inertia(self, I: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Transform spatial inertia by spatial transform."""
        return X.T @ I @ X

    def _compute_spatial_transform(self, pos1, quat1, pos2, quat2) -> np.ndarray:
        """Compute 6x6 spatial transform between frames."""
        # Simplified - full implementation needed
        return np.eye(6)

    def _get_body_acc(self, body_idx: int) -> np.ndarray:
        """Get spatial acceleration of body."""
        return np.zeros(6)  # Placeholder
```

### 3. Collision Detection System

```python
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

class BroadPhase:
    """Broad-phase collision detection using AABB tree."""

    def __init__(self):
        self.aabb_tree = None

    def find_overlapping_pairs(self, model: Model, state: State) -> List[Tuple[int, int]]:
        """
        Find all potentially overlapping geometry pairs.

        Returns:
            List of (geom1_idx, geom2_idx) pairs to check
        """
        pairs = []

        # Simple O(n^2) for now - use AABB tree for efficiency
        geom_aabbs = []
        geom_to_body = []

        geom_idx = 0
        for body_idx, body in enumerate(model.bodies):
            for geom in body.geoms:
                aabb_min, aabb_max = geom.get_aabb(
                    state.xpos[body_idx], state.xquat[body_idx]
                )
                geom_aabbs.append((aabb_min, aabb_max))
                geom_to_body.append(body_idx)
                geom_idx += 1

        # Check all pairs
        n_geoms = len(geom_aabbs)
        for i in range(n_geoms):
            for j in range(i + 1, n_geoms):
                # Skip same body
                if geom_to_body[i] == geom_to_body[j]:
                    continue

                # Check AABB overlap
                if self._aabb_overlap(geom_aabbs[i], geom_aabbs[j]):
                    pairs.append((i, j))

        return pairs

    @staticmethod
    def _aabb_overlap(aabb1: Tuple, aabb2: Tuple) -> bool:
        """Check if two AABBs overlap."""
        min1, max1 = aabb1
        min2, max2 = aabb2

        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])

class NarrowPhase:
    """Narrow-phase collision detection for exact contact computation."""

    def check_collision(self, geom1: Geom, pos1: np.ndarray, quat1: np.ndarray,
                        geom2: Geom, pos2: np.ndarray, quat2: np.ndarray) -> Optional[Contact]:
        """
        Check collision between two geometries.

        Returns:
            Contact if collision, None otherwise
        """
        # Dispatch based on geometry types
        type_pair = (geom1.geom_type, geom2.geom_type)

        if type_pair == (GeomType.SPHERE, GeomType.SPHERE):
            return self._sphere_sphere(geom1, pos1, geom2, pos2)

        elif type_pair == (GeomType.SPHERE, GeomType.PLANE):
            return self._sphere_plane(geom1, pos1, geom2, pos2, quat2)

        elif type_pair == (GeomType.PLANE, GeomType.SPHERE):
            contact = self._sphere_plane(geom2, pos2, geom1, pos1, quat1)
            if contact:
                contact.normal = -contact.normal
            return contact

        elif type_pair == (GeomType.SPHERE, GeomType.BOX):
            return self._sphere_box(geom1, pos1, geom2, pos2, quat2)

        elif type_pair == (GeomType.CAPSULE, GeomType.PLANE):
            return self._capsule_plane(geom1, pos1, quat1, geom2, pos2, quat2)

        elif type_pair == (GeomType.CAPSULE, GeomType.CAPSULE):
            return self._capsule_capsule(geom1, pos1, quat1, geom2, pos2, quat2)

        else:
            # Use GJK/EPA for general case
            return self._gjk_epa(geom1, pos1, quat1, geom2, pos2, quat2)

    def _sphere_sphere(self, geom1: Geom, pos1: np.ndarray,
                       geom2: Geom, pos2: np.ndarray) -> Optional[Contact]:
        """Sphere-sphere collision."""
        r1, r2 = geom1.size[0], geom2.size[0]
        d = pos2 - pos1
        dist = np.linalg.norm(d)

        if dist < r1 + r2:
            normal = d / (dist + 1e-8)
            penetration = r1 + r2 - dist
            contact_pos = pos1 + normal * (r1 - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,  # Set by caller
                body1_idx=0, body2_idx=0,
                friction=min(geom1.friction, geom2.friction),
                restitution=max(geom1.restitution, geom2.restitution)
            )
        return None

    def _sphere_plane(self, sphere: Geom, sphere_pos: np.ndarray,
                      plane: Geom, plane_pos: np.ndarray,
                      plane_quat: np.ndarray) -> Optional[Contact]:
        """Sphere-plane collision."""
        # Plane normal in world frame
        normal = ForwardDynamics._quat_rotate(plane_quat, np.array([0, 0, 1]))

        # Distance from sphere center to plane
        d = np.dot(sphere_pos - plane_pos, normal)
        r = sphere.size[0]

        if d < r:
            contact_pos = sphere_pos - normal * d
            penetration = r - d

            return Contact(
                pos=contact_pos,
                normal=-normal,  # Point into sphere
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(sphere.friction, plane.friction),
                restitution=max(sphere.restitution, plane.restitution)
            )
        return None

    def _sphere_box(self, sphere: Geom, sphere_pos: np.ndarray,
                    box: Geom, box_pos: np.ndarray,
                    box_quat: np.ndarray) -> Optional[Contact]:
        """Sphere-box collision."""
        r = sphere.size[0]
        half = box.size

        # Transform sphere center to box local frame
        d = sphere_pos - box_pos
        q_inv = np.array([box_quat[0], -box_quat[1], -box_quat[2], -box_quat[3]])
        local_pos = ForwardDynamics._quat_rotate(q_inv, d)

        # Find closest point on box
        closest = np.clip(local_pos, -half, half)

        # Distance to closest point
        diff = local_pos - closest
        dist = np.linalg.norm(diff)

        if dist < r:
            if dist > 1e-8:
                normal_local = diff / dist
            else:
                # Sphere center inside box
                # Find closest face
                face_dist = np.minimum(half - local_pos, local_pos + half)
                min_axis = np.argmin(face_dist)
                normal_local = np.zeros(3)
                normal_local[min_axis] = np.sign(local_pos[min_axis])

            normal = ForwardDynamics._quat_rotate(box_quat, normal_local)
            penetration = r - dist
            contact_pos = sphere_pos - normal * (r - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(sphere.friction, box.friction),
                restitution=max(sphere.restitution, box.restitution)
            )
        return None

    def _capsule_plane(self, capsule: Geom, cap_pos: np.ndarray, cap_quat: np.ndarray,
                       plane: Geom, plane_pos: np.ndarray,
                       plane_quat: np.ndarray) -> Optional[Contact]:
        """Capsule-plane collision."""
        r = capsule.size[0]
        half_len = capsule.size[1]

        # Capsule endpoints
        axis = ForwardDynamics._quat_rotate(cap_quat, np.array([0, 0, 1]))
        p1 = cap_pos - axis * half_len
        p2 = cap_pos + axis * half_len

        # Plane normal
        normal = ForwardDynamics._quat_rotate(plane_quat, np.array([0, 0, 1]))

        # Check both endpoints
        contacts = []
        for p in [p1, p2]:
            d = np.dot(p - plane_pos, normal)
            if d < r:
                contact_pos = p - normal * d
                contacts.append(Contact(
                    pos=contact_pos,
                    normal=-normal,
                    penetration=r - d,
                    geom1_idx=0, geom2_idx=0,
                    body1_idx=0, body2_idx=0,
                    friction=min(capsule.friction, plane.friction),
                    restitution=max(capsule.restitution, plane.restitution)
                ))

        # Return deepest contact
        if contacts:
            return max(contacts, key=lambda c: c.penetration)
        return None

    def _capsule_capsule(self, cap1: Geom, pos1: np.ndarray, quat1: np.ndarray,
                         cap2: Geom, pos2: np.ndarray, quat2: np.ndarray) -> Optional[Contact]:
        """Capsule-capsule collision."""
        # Find closest points between line segments
        r1, h1 = cap1.size[0], cap1.size[1]
        r2, h2 = cap2.size[0], cap2.size[1]

        axis1 = ForwardDynamics._quat_rotate(quat1, np.array([0, 0, 1]))
        axis2 = ForwardDynamics._quat_rotate(quat2, np.array([0, 0, 1]))

        a1 = pos1 - axis1 * h1
        b1 = pos1 + axis1 * h1
        a2 = pos2 - axis2 * h2
        b2 = pos2 + axis2 * h2

        # Closest points on line segments
        c1, c2 = self._closest_points_segments(a1, b1, a2, b2)

        dist = np.linalg.norm(c2 - c1)
        if dist < r1 + r2:
            normal = (c2 - c1) / (dist + 1e-8)
            penetration = r1 + r2 - dist
            contact_pos = c1 + normal * (r1 - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(cap1.friction, cap2.friction),
                restitution=max(cap1.restitution, cap2.restitution)
            )
        return None

    @staticmethod
    def _closest_points_segments(a1, b1, a2, b2):
        """Find closest points between two line segments."""
        d1 = b1 - a1
        d2 = b2 - a2
        r = a1 - a2

        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d1, r)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        denom = a * e - b * b
        if abs(denom) < 1e-8:
            # Parallel segments
            s = 0
            t = f / e if e > 1e-8 else 0
        else:
            s = (b * f - c * e) / denom
            t = (a * f - b * c) / denom

        # Clamp to segment
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)

        c1 = a1 + s * d1
        c2 = a2 + t * d2

        return c1, c2

    def _gjk_epa(self, geom1, pos1, quat1, geom2, pos2, quat2) -> Optional[Contact]:
        """GJK + EPA for general convex collision."""
        # Implement GJK for intersection test
        # Implement EPA for penetration depth
        return None  # Placeholder

class CollisionSystem:
    """Complete collision detection system."""

    def __init__(self, model: Model):
        self.model = model
        self.broad_phase = BroadPhase()
        self.narrow_phase = NarrowPhase()

    def detect_contacts(self, state: State) -> List[Contact]:
        """Detect all contacts in current state."""
        contacts = []

        # Broad phase
        pairs = self.broad_phase.find_overlapping_pairs(self.model, state)

        # Narrow phase
        geom_idx = 0
        geom_map = []  # (body_idx, geom_idx within body)
        for body_idx, body in enumerate(self.model.bodies):
            for local_idx, geom in enumerate(body.geoms):
                geom_map.append((body_idx, local_idx))

        for i, j in pairs:
            body1_idx, local1 = geom_map[i]
            body2_idx, local2 = geom_map[j]

            geom1 = self.model.bodies[body1_idx].geoms[local1]
            geom2 = self.model.bodies[body2_idx].geoms[local2]

            contact = self.narrow_phase.check_collision(
                geom1, state.xpos[body1_idx], state.xquat[body1_idx],
                geom2, state.xpos[body2_idx], state.xquat[body2_idx]
            )

            if contact:
                contact.geom1_idx = i
                contact.geom2_idx = j
                contact.body1_idx = body1_idx
                contact.body2_idx = body2_idx
                contacts.append(contact)

        return contacts
```

### 4. Constraint Solver

```python
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
        # J * dv = -bias
        # where J is contact Jacobian

        J = np.zeros((n_contacts * 3, self.model.nv))  # Normal + 2 friction
        bias = np.zeros(n_contacts * 3)
        bounds = np.zeros((n_contacts * 3, 2))  # (lower, upper) for each constraint

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

            # Friction cone bounds
            mu = contact.friction
            bounds[i*3+1] = (-mu, mu)  # Will be scaled by normal
            bounds[i*3+2] = (-mu, mu)

        # Compute effective mass matrix
        # M_eff = J * M^{-1} * J^T
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

        v1_n = n @ (body1_J[:3] + np.cross(body1_J[3:].T, r1).T)
        v1_t1 = t1 @ (body1_J[:3] + np.cross(body1_J[3:].T, r1).T)
        v1_t2 = t2 @ (body1_J[:3] + np.cross(body1_J[3:].T, r1).T)

        J_n += v1_n
        J_t1 += v1_t1
        J_t2 += v1_t2

        # Contributions from body 2 (negative because relative)
        if contact.body2_idx >= 0:
            r2 = contact.pos - state.xpos[contact.body2_idx]
            body2_J = self._body_jacobian(state, contact.body2_idx)

            v2_n = n @ (body2_J[:3] + np.cross(body2_J[3:].T, r2).T)
            v2_t1 = t1 @ (body2_J[:3] + np.cross(body2_J[3:].T, r2).T)
            v2_t2 = t2 @ (body2_J[:3] + np.cross(body2_J[3:].T, r2).T)

            J_n -= v2_n
            J_t1 -= v2_t1
            J_t2 -= v2_t2

        return J_n, J_t1, J_t2

    def _body_jacobian(self, state: State, body_idx: int) -> np.ndarray:
        """Compute Jacobian mapping joint velocities to body velocity."""
        # Simplified - return identity for demonstration
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
                    J[:, start:end] += S

            current = body.parent

        return J

    def _inverse_mass_matrix(self, state: State) -> np.ndarray:
        """Compute inverse of mass matrix."""
        # Simplified - use diagonal approximation
        M_inv = np.eye(self.model.nv)
        return M_inv

    def _perpendicular(self, v: np.ndarray) -> np.ndarray:
        """Get a vector perpendicular to v."""
        if abs(v[0]) < 0.9:
            p = np.array([1, 0, 0])
        else:
            p = np.array([0, 1, 0])
        return np.cross(v, p)

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
        else:
            return np.zeros((6, 0))
```

### 5. Numerical Integrator

```python
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

        # Update positions using new velocity
        new_state.qpos = state.qpos + new_state.qvel * dt

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
        # For free joints
        for joint in self.model.joints:
            if joint.joint_type == JointType.FREE:
                q_idx = joint.qpos_idx + 3  # Skip position
                q = state.qpos[q_idx:q_idx+4]
                state.qpos[q_idx:q_idx+4] = q / np.linalg.norm(q)
```

### 6. RL Environment API

```python
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

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = State.create(self.model)

        # Randomization
        self._apply_domain_randomization()

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in environment.

        Args:
            action: Control inputs (nu,)

        Returns:
            observation, reward, done, info
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Step simulation
        self.state = self.integrator.step(self.state, action)

        # Compute observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self._check_termination()

        info = {
            'time': self.state.time,
            'n_contacts': len(self.state.contacts)
        }

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation from state."""
        obs_parts = []

        # Joint positions
        obs_parts.append(self.state.qpos)

        # Joint velocities
        obs_parts.append(self.state.qvel)

        # Body positions (optional)
        obs_parts.append(self.state.xpos.flatten())

        return np.concatenate(obs_parts)

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        # Override in subclass
        return 0.0

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Override in subclass
        return False

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension."""
        return (self.model.nq + self.model.nv +
                self.model.nbody * 3)

    def _apply_domain_randomization(self):
        """Apply domain randomization for sim2real."""
        # Mass randomization
        for body in self.model.bodies:
            body.inertia.mass *= np.random.uniform(0.9, 1.1)

        # Friction randomization
        for body in self.model.bodies:
            for geom in body.geoms:
                geom.friction *= np.random.uniform(0.8, 1.2)

class BatchedEnvironment:
    """
    Batched environment for parallel simulation.
    """

    def __init__(self, model: Model, num_envs: int):
        self.num_envs = num_envs
        self.envs = [PhysicsEnvironment(model) for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        obs = np.stack([env.reset() for env in self.envs])
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Step all environments in parallel."""
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]

        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        return obs, rewards, dones, infos
```

## Implementation Phases

### Phase 1: Rigid Body Foundation (Weeks 1-3)
- [ ] Body, Joint, Geom data structures
- [ ] Model class with configuration
- [ ] State representation
- [ ] Forward kinematics
- [ ] Basic quaternion operations

### Phase 2: Forward Dynamics (Weeks 4-6)
- [ ] Mass matrix computation
- [ ] Recursive Newton-Euler algorithm
- [ ] Articulated Body Algorithm
- [ ] Joint force computation
- [ ] Gravity and external forces

### Phase 3: Collision Detection (Weeks 7-9)
- [ ] AABB broad phase
- [ ] Sphere-sphere collision
- [ ] Sphere-plane collision
- [ ] Box collisions
- [ ] Capsule collisions
- [ ] GJK/EPA for general convex

### Phase 4: Constraint Solving (Weeks 10-12)
- [ ] Contact Jacobian computation
- [ ] Projected Gauss-Seidel solver
- [ ] Friction cone constraints
- [ ] Warmstarting
- [ ] Joint limit constraints

### Phase 5: Integration (Weeks 13-14)
- [ ] Euler integrator
- [ ] Semi-implicit Euler
- [ ] RK4 integrator
- [ ] Quaternion normalization
- [ ] Time stepping loop

### Phase 6: RL Environment (Weeks 15-16)
- [ ] Gym-compatible API
- [ ] Observation extraction
- [ ] Reward functions
- [ ] Domain randomization
- [ ] Batched environments

### Phase 7: GPU Acceleration (Weeks 17-20)
- [ ] CUDA/Taichi port
- [ ] Parallel collision detection
- [ ] Batched dynamics
- [ ] Differentiable physics (stretch)
- [ ] Performance optimization

## Testing Strategy

### Unit Tests
```python
class TestRigidBody:
    def test_inertia_box(self):
        """Test box inertia computation."""
        inertia = Inertia.from_box(1.0, np.array([1, 2, 3]))
        # Check principal moments
        assert inertia.inertia[0, 0] == pytest.approx((4 + 36) / 12)

    def test_quaternion_rotation(self):
        """Test quaternion rotation."""
        q = np.array([0.707, 0.707, 0, 0])  # 90 deg about x
        v = np.array([0, 1, 0])
        result = ForwardDynamics._quat_rotate(q, v)
        assert result == pytest.approx([0, 0, 1])

class TestCollision:
    def test_sphere_sphere(self):
        """Test sphere-sphere collision."""
        geom1 = Geom(GeomType.SPHERE, np.array([1.0]))
        geom2 = Geom(GeomType.SPHERE, np.array([1.0]))

        narrow = NarrowPhase()
        contact = narrow._sphere_sphere(
            geom1, np.array([0, 0, 0]),
            geom2, np.array([1.5, 0, 0])
        )

        assert contact is not None
        assert contact.penetration == pytest.approx(0.5)

class TestIntegration:
    def test_free_fall(self):
        """Test free fall under gravity."""
        model = Model()
        # ... setup model

        state = State.create(model)
        integrator = Integrator(model)

        # Step for 1 second
        for _ in range(500):
            state = integrator.step(state, np.zeros(model.nu))

        # Check final position
        expected_drop = 0.5 * 9.81 * 1.0**2
        # ... verify position
```

### Integration Tests
```python
class TestPendulum:
    def test_pendulum_period(self):
        """Test pendulum oscillation period."""
        # Create pendulum model
        model = create_pendulum_model(length=1.0)

        # Start from horizontal
        state = State.create(model)
        state.qpos[0] = np.pi / 2

        integrator = Integrator(model)

        # Simulate and find period
        # T = 2*pi*sqrt(L/g)
```

## Performance Targets

| Benchmark | Target | Notes |
|-----------|--------|-------|
| Single body step | 10 us | Forward dynamics only |
| 100-body step | 1 ms | With contacts |
| 1000 parallel envs | 10 ms | Batched simulation |
| GPU 4096 envs | 5 ms | With contacts |

## Dependencies

- NumPy
- numpy-quaternion (optional, can implement)
- Numba (for JIT acceleration)
- CUDA/Taichi (for GPU)

## References

- Featherstone: Rigid Body Dynamics Algorithms
- MuJoCo: A physics engine for model-based control
- Bullet Physics
- PyBullet
