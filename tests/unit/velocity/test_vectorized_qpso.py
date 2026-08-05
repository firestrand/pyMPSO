import numpy as np

from src.particles.swarm_state import SwarmState
from src.velocity.config import QuantumPSOVelocityConfig
from src.velocity.vectorized_qpso import VectorizedQPSOVelocityUpdate


def test_vectorized_qpso_velocity_update():
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    velocities = np.array([[0.5, 0.5], [0.0, 0.0], [-0.5, -0.5]])
    pbest_positions = np.array([[1.0, 1.0], [1.0, 1.0], [-2.0, -2.0]])
    informant_matrix = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=np.zeros(3),
        current_fitness=np.zeros(3),
    )
    rng_seed = 2026
    rng = np.random.RandomState(rng_seed)

    config = QuantumPSOVelocityConfig(beta=1.0, rng=rng)
    strategy = VectorizedQPSOVelocityUpdate()

    new_velocities = strategy.update_velocities(state, informant_matrix, config)

    mbest = np.mean(pbest_positions, axis=0, keepdims=True)
    expected_rng = np.random.RandomState(rng_seed)
    attractor_weights = expected_rng.random(positions.shape)
    rand_u = np.clip(expected_rng.random(positions.shape), 1e-12, 1.0)
    local_best = attractor_weights * pbest_positions + (1.0 - attractor_weights) * mbest
    expected_step = np.abs(local_best - positions) * np.log(1.0 / rand_u)
    random_sign = np.where(expected_rng.random(positions.shape) < 0.5, 1.0, -1.0)

    expected_velocities = random_sign * expected_step

    np.testing.assert_array_almost_equal(new_velocities, expected_velocities)


def test_vectorized_qpso_velocity_update_decay():
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    velocities = np.array([[0.5, 0.5], [0.0, 0.0], [-0.5, -0.5]])
    pbest_positions = np.array([[1.0, 1.0], [1.0, 1.0], [-2.0, -2.0]])
    informant_matrix = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=np.zeros(3),
        current_fitness=np.zeros(3),
    )

    current_iteration = 5
    max_iterations = 10
    rng_seed = 2026
    rng = np.random.RandomState(rng_seed)

    config = QuantumPSOVelocityConfig(
        beta=1.0,
        beta_min=0.5,
        beta_max=1.5,
        rng=rng,
    )
    strategy = VectorizedQPSOVelocityUpdate()

    new_velocities = strategy.update_velocities(
        state,
        informant_matrix,
        config,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
    )

    ratio = min(max(float(current_iteration) / float(max_iterations), 0.0), 1.0)
    beta = 1.5 + (0.5 - 1.5) * ratio

    mbest = np.mean(pbest_positions, axis=0, keepdims=True)
    expected_rng = np.random.RandomState(rng_seed)
    attractor_weights = expected_rng.random(positions.shape)
    rand_u = np.clip(expected_rng.random(positions.shape), 1e-12, 1.0)
    local_best = attractor_weights * pbest_positions + (1.0 - attractor_weights) * mbest
    expected_step = beta * np.abs(local_best - positions) * np.log(1.0 / rand_u)
    random_sign = np.where(expected_rng.random(positions.shape) < 0.5, 1.0, -1.0)

    expected_velocities = random_sign * expected_step

    np.testing.assert_array_almost_equal(new_velocities, expected_velocities)
