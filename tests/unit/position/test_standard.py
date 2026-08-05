import numpy as np

from src.particles.swarm_state import SwarmState
from src.position.standard import StandardPositionUpdate


def test_standard_position_update():
    """Verify standard position update correctly adds velocity to position."""
    n_particles = 3

    positions = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])

    velocities = np.array([[0.5, -0.5], [1.0, 2.0], [-0.5, 1.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=np.zeros_like(positions),
        pbest_fitness=np.zeros(n_particles),
        current_fitness=np.zeros(n_particles),
    )

    strategy = StandardPositionUpdate()
    new_positions = strategy.update_positions(state)

    expected_positions = np.array([[1.5, 1.5], [4.0, 6.0], [-1.5, -1.0]])

    np.testing.assert_array_equal(new_positions, expected_positions)
    # Ensure original state is not modified if we don't want it to be.
    # Or, actually, in a vectorized engine, we usually modify in-place or return new array.
    # The protocol says "Calculates and returns the new positions for the swarm."
    # Let's ensure it returns the correct values.
