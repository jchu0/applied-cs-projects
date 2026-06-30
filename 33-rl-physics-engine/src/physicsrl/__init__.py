"""PhysicsRL - MuJoCo-lite physics engine for reinforcement learning."""

from .core import (
    GeomType,
    JointType,
    Inertia,
    Geom,
    Joint,
    Actuator,
    Body,
    Model,
    State,
    Contact,
)
from .dynamics import ForwardDynamics
from .collision import BroadPhase, NarrowPhase, CollisionSystem
from .solver import ConstraintSolver
from .integration import Integrator
from .environment import PhysicsEnvironment, BatchedEnvironment

__version__ = "0.1.0"

__all__ = [
    # Core types
    "GeomType",
    "JointType",
    "Inertia",
    "Geom",
    "Joint",
    "Actuator",
    "Body",
    "Model",
    "State",
    "Contact",
    # Dynamics
    "ForwardDynamics",
    # Collision
    "BroadPhase",
    "NarrowPhase",
    "CollisionSystem",
    # Solver
    "ConstraintSolver",
    # Integration
    "Integrator",
    # Environment
    "PhysicsEnvironment",
    "BatchedEnvironment",
]
