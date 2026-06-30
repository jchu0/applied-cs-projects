# RL Physics Engine

Differentiable physics engine for reinforcement learning with rigid body dynamics and constraint solving.

## Features

- **Rigid Body Dynamics**: Position, velocity, acceleration simulation
- **Collision Detection**: Sphere, box, plane primitives
- **Constraints**: Joints, contacts, friction
- **Differentiable**: Gradients through physics simulation
- **NumPy Backend**: Pure Python, no external dependencies

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from physicsrl import World, RigidBody, Sphere

# Create world with gravity
world = World(gravity=[0, -9.81, 0], dt=0.01)

# Add a falling sphere
sphere = RigidBody(
    shape=Sphere(radius=0.5),
    mass=1.0,
    position=[0, 10, 0]
)
world.add_body(sphere)

# Simulate
for _ in range(1000):
    world.step()
    print(f"Position: {sphere.position}")
```

## Differentiable Physics

```python
from physicsrl import grad_through_physics

# Define reward based on final position
def reward_fn(initial_velocity):
    world = create_world()
    world.bodies[0].velocity = initial_velocity

    for _ in range(100):
        world.step()

    return -world.bodies[0].position[0]  # Maximize x distance

# Compute gradient of reward w.r.t. initial velocity
grad_reward = grad_through_physics(reward_fn)
velocity_grad = grad_reward([1.0, 1.0, 0.0])
```

## Components

| Component | Description |
|-----------|-------------|
| `World` | Physics simulation container |
| `RigidBody` | Dynamic object with mass |
| `Sphere/Box/Plane` | Collision shapes |
| `Joint` | Constraint between bodies |

## Testing

```bash
pytest tests/ -v  # 143 tests
```
