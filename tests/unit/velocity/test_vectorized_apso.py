import numpy as np

from src.particles.swarm_state import SwarmState
from src.velocity.config import AdaptivePSOVelocityConfig
from src.velocity.vectorized_apso import VectorizedAdaptivePSOVelocityUpdate


def test_vectorized_apso_velocity_update_uses_schedule():
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    velocities = np.array([[0.5, 0.5], [0.0, 0.0], [-0.5, -0.5]])
    pbest_positions = np.array([[1.0, 1.0], [1.0, 1.0], [-2.0, -2.0]])
    informant_matrix = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    current_iteration = 5
    max_iterations = 10
    rng_seed = 2026
    rng = np.random.RandomState(rng_seed)

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=np.zeros(3),
        current_fitness=np.zeros(3),
    )

    config = AdaptivePSOVelocityConfig(
        w_max=1.0,
        w_min=0.0,
        c1_max=2.0,
        c1_min=0.0,
        c2_min=0.0,
        c2_max=2.0,
        rng=rng,
    )

    strategy = VectorizedAdaptivePSOVelocityUpdate()
    new_velocities = strategy.update_velocities(
        state,
        informant_matrix,
        config,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
    )

    expected_ratio = min(
        max(float(current_iteration) / float(max_iterations), 0.0),
        1.0,
    )
    inertia = 1.0 + (0.0 - 1.0) * expected_ratio
    c1 = 2.0 + (0.0 - 2.0) * expected_ratio
    c2 = 0.0 + (2.0 - 0.0) * expected_ratio

    expected_rng = np.random.RandomState(rng_seed)
    r1 = expected_rng.random(positions.shape)
    r2 = expected_rng.random(positions.shape)
    expected = inertia * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (informant_matrix - positions)

    np.testing.assert_array_almost_equal(new_velocities, expected)
