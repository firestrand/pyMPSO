import numpy as np

from src.boundary.vectorized_position_clamping import VectorizedPositionClampingBoundaryHandler
from src.particles.swarm_state import SwarmState


def test_vectorized_position_clamping_boundary():
    """Verify out-of-bounds positions are clamped and velocities zeroed."""
    n_particles = 3

    positions = np.array(
        [
            [0.0, 0.0],
            [15.0, 15.0],  # Out of bounds on max side
            [-15.0, -15.0],  # Out of bounds on min side
        ]
    )

    velocities = np.array([[1.0, 1.0], [2.0, 2.0], [-2.0, -2.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=np.zeros_like(positions),
        pbest_fitness=np.zeros(n_particles),
        current_fitness=np.zeros(n_particles),
    )

    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])

    handler = VectorizedPositionClampingBoundaryHandler()
    handler.apply(state, bounds)

    # Expected positions:
    expected_positions = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]])

    # Expected velocities:
    expected_velocities = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    np.testing.assert_array_equal(state.positions, expected_positions)
    np.testing.assert_array_equal(state.velocities, expected_velocities)


def test_vectorized_position_clamping_boundary_no_bounds():
    """Verify handler does nothing when bounds are None."""
    n_particles = 1

    positions = np.array([[15.0, 15.0]])
    velocities = np.array([[2.0, 2.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=np.zeros_like(positions),
        pbest_fitness=np.zeros(n_particles),
        current_fitness=np.zeros(n_particles),
    )

    handler = VectorizedPositionClampingBoundaryHandler()
    handler.apply(state, None)

    # State should remain unchanged
    np.testing.assert_array_equal(state.positions, np.array([[15.0, 15.0]]))
    np.testing.assert_array_equal(state.velocities, np.array([[2.0, 2.0]]))
