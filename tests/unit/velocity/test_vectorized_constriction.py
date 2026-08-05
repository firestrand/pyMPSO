import numpy as np

from src.particles.swarm_state import SwarmState
from src.velocity.config import ConstrictionVelocityConfig
from src.velocity.vectorized_constriction import VectorizedConstrictionVelocityUpdate


def test_vectorized_constriction_velocity_update():
    """Verify vectorized constriction velocity update correctly applies the formula."""
    n_particles = 3

    positions = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])

    velocities = np.array([[0.5, 0.5], [0.0, 0.0], [-0.5, -0.5]])

    pbest_positions = np.array([[1.0, 1.0], [1.0, 1.0], [-2.0, -2.0]])

    informant_matrix = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=pbest_positions,
        pbest_fitness=np.zeros(n_particles),
        current_fitness=np.zeros(n_particles),
    )

    class MockRNG:
        def random(self, size):
            return np.full(size, 0.5)

    config = ConstrictionVelocityConfig(phi1=2.05, phi2=2.05, rng=MockRNG())  # type: ignore

    strategy = VectorizedConstrictionVelocityUpdate()

    chi = strategy._calculate_constriction_coefficient(2.05, 2.05)
    # Expected chi ≈ 0.7298

    new_velocities = strategy.update_velocities(state, informant_matrix, config)

    # Formula: chi * (v(t) + phi1*r1*(pbest - x) + phi2*r2*(informant - x))
    # phi1*r1 = 2.05*0.5 = 1.025
    # phi2*r2 = 2.05*0.5 = 1.025

    # Particle 0:
    # inner = [0.5, 0.5] + 1.025*([1,1]) + 1.025*([2,2])
    # inner = [0.5, 0.5] + [1.025, 1.025] + [2.05, 2.05] = [3.575, 3.575]
    # v = chi * [3.575, 3.575]

    expected_p0 = chi * 3.575

    assert np.allclose(new_velocities[0], np.array([expected_p0, expected_p0]))
    assert new_velocities.shape == (3, 2)
