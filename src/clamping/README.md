# Clamping Module

The Clamping module provides abstractions and implementations for restricting values (positions, velocities) within defined boundaries in the PSO algorithm. This module serves as a central place for all clamping-related functionality, providing a clean separation of concerns.

## Structure

The module is organized into three main components:

1. **Base Abstractions** (`base.py`):
   - `ClampingStrategy`: The core abstract base class for all clamping strategies
   - `PositionClampingStrategy`: For position-specific clamping behaviors
   - `VelocityClampingStrategy`: For velocity-specific clamping behaviors

2. **Position Clamping** (`position.py`):
   - `BasicPositionClampingStrategy`: Simple clip-to-bounds approach
   - `VelocityResetPositionClampingStrategy`: Clips positions and provides mask for velocity reset
   - `apply_position_clamping()`: Helper function to apply position clamping to particles

3. **Velocity Clamping** (`velocity.py`):
   - `MaxNormVelocityClampingStrategy`: Clamps velocity to fixed maximum value
   - `RelativeBoundsVelocityClampingStrategy`: Clamps velocity as proportion of search space
   - `NoClampingStrategy`: Passthrough strategy for when no clamping is desired

## Usage

### Velocity Clamping

Velocity clamping strategies are used by velocity update strategies:

```python
from src.clamping import RelativeBoundsVelocityClampingStrategy
from src.velocity import InertiaWeightVelocityUpdate

# Create a velocity clamping strategy
velocity_clamping = RelativeBoundsVelocityClampingStrategy(max_velocity_ratio=0.1)

# Use it with a velocity update strategy
velocity_update = InertiaWeightVelocityUpdate(clamping_strategy=velocity_clamping)
```

### Position Clamping

Position clamping strategies are used for boundary handling:

```python
from src.clamping import VelocityResetPositionClampingStrategy, apply_position_clamping
from src.particles import Particle

# Create a position clamping strategy
position_strategy = VelocityResetPositionClampingStrategy()

# Apply it to a particle
particle = Particle(...)
bounds = np.array([[-5, 5], [-5, 5]])  # Min/max bounds for each dimension
apply_position_clamping(particle=particle, bounds=bounds, position_strategy=position_strategy, reset_velocity=True)
```

## Integration with Boundary Handling

The `PositionClampingBoundaryHandler` in the `boundary` module now uses the position clamping strategies from this module:

```python
from src.boundary.position_clamping import PositionClampingBoundaryHandler

# Create a boundary handler that uses position clamping
boundary_handler = PositionClampingBoundaryHandler(reset_velocity=True)

# Apply it to a particle
boundary_handler.apply(particle, bounds)
```

## Design Benefits

1. **Separation of Concerns**: Clamping is now a distinct concept, not tied to either velocity update or boundary handling
2. **Reusability**: The same clamping strategies can be used across different parts of the algorithm
3. **Extensibility**: New clamping strategies can be added by implementing one of the abstract base classes
4. **Testability**: Each clamping strategy can be tested in isolation 