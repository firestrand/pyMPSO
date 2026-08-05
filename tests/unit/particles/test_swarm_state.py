import numpy as np
import pytest

from src.particles.swarm_state import SwarmState


def test_swarm_state_initialization():
    """Verify SwarmState initializes correctly with valid NumPy arrays."""
    n_particles = 10
    dimensions = 5

    positions = np.zeros((n_particles, dimensions))
    velocities = np.ones((n_particles, dimensions))
    pbest_positions = np.copy(positions)
    pbest_fitness = np.full(n_particles, np.inf)
    current_fitness = np.full(n_particles, np.inf)

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=pbest_fitness,
        current_fitness=current_fitness,
    )

    assert state.num_particles == 10
    assert state.dimensions == 5
    np.testing.assert_array_equal(state.positions, positions)
    np.testing.assert_array_equal(state.velocities, velocities)
    np.testing.assert_array_equal(state.pbest_positions, pbest_positions)
    np.testing.assert_array_equal(state.pbest_fitness, pbest_fitness)
    np.testing.assert_array_equal(state.current_fitness, current_fitness)


def test_swarm_state_shape_validation():
    """Verify SwarmState raises ValueError on inconsistent array shapes."""
    n_particles = 5
    dimensions = 3

    positions = np.zeros((n_particles, dimensions))
    valid_velocities = np.zeros((n_particles, dimensions))
    valid_pbest_pos = np.zeros((n_particles, dimensions))
    valid_pbest_fit = np.zeros(n_particles)
    valid_curr_fit = np.zeros(n_particles)

    # Test invalid positions dimensionality
    with pytest.raises(ValueError, match="positions must be a 2D array"):
        SwarmState(
            positions=np.zeros(15),  # 1D array
            velocities=valid_velocities,
            pbest_positions=valid_pbest_pos,
            pbest_fitness=valid_pbest_fit,
            current_fitness=valid_curr_fit,
        )

    # Test invalid velocities shape
    with pytest.raises(ValueError, match="velocities shape mismatch"):
        SwarmState(
            positions=positions,
            velocities=np.zeros((n_particles, dimensions + 1)),
            pbest_positions=valid_pbest_pos,
            pbest_fitness=valid_pbest_fit,
            current_fitness=valid_curr_fit,
        )

    # Test invalid pbest_positions shape
    with pytest.raises(ValueError, match="pbest_positions shape mismatch"):
        SwarmState(
            positions=positions,
            velocities=valid_velocities,
            pbest_positions=np.zeros((n_particles, dimensions + 1)),
            pbest_fitness=valid_pbest_fit,
            current_fitness=valid_curr_fit,
        )

    # Test invalid pbest_fitness shape
    with pytest.raises(ValueError, match="pbest_fitness shape mismatch"):
        SwarmState(
            positions=positions,
            velocities=valid_velocities,
            pbest_positions=valid_pbest_pos,
            pbest_fitness=np.zeros(n_particles + 1),
            current_fitness=valid_curr_fit,
        )

    # Test invalid current_fitness shape
    with pytest.raises(ValueError, match="current_fitness shape mismatch"):
        SwarmState(
            positions=positions,
            velocities=valid_velocities,
            pbest_positions=valid_pbest_pos,
            pbest_fitness=valid_pbest_fit,
            current_fitness=np.zeros(n_particles + 1),
        )


def test_get_global_best():
    """Verify get_global_best returns the correct position and fitness."""
    n_particles = 3
    dimensions = 2

    positions = np.zeros((n_particles, dimensions))
    velocities = np.zeros((n_particles, dimensions))

    pbest_positions = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],  # Best position
            [3.0, 3.0],
        ]
    )
    pbest_fitness = np.array([10.0, 5.0, 15.0])  # 5.0 is the best (minimum)
    current_fitness = np.zeros(n_particles)

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=pbest_fitness,
        current_fitness=current_fitness,
    )

    best_pos, best_fit = state.get_global_best()

    assert best_fit == 5.0
    np.testing.assert_array_equal(best_pos, np.array([2.0, 2.0]))

    # Ensure it's a copy
    best_pos[0] = 99.0
    assert state.pbest_positions[1][0] == 2.0
