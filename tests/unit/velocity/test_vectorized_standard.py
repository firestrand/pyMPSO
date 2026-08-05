import numpy as np

from src.particles.swarm_state import SwarmState
from src.velocity.config import StandardVelocityConfig
from src.velocity.vectorized_standard import VectorizedStandardVelocityUpdate


def test_vectorized_standard_velocity_update():
    """Verify vectorized standard velocity update correctly applies the formula."""
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

    # Use deterministic RNG for testing
    # rng = np.random.RandomState(42)
    # The RNG will generate two arrays of shape (3, 2).
    # Since we need exactly predictable values, let's inject a custom RNG or monkeypatch if needed,
    # but we can also just run it and check the formula.
    # Actually, we can pre-calculate expected based on the exact RNG sequence, or we can mock the RNG.

    class MockRNG:
        def random(self, size):
            # Return an array of 0.5s for predictable cognitive and social components
            return np.full(size, 0.5)

    config = StandardVelocityConfig(c1=2.0, c2=2.0, rng=MockRNG())  # type: ignore

    strategy = VectorizedStandardVelocityUpdate()
    new_velocities = strategy.update_velocities(state, informant_matrix, config)

    # v(t+1) = v(t) + c1 * r1 * (pbest - x) + c2 * r2 * (informant - x)
    # Since r1 = 0.5, r2 = 0.5, c1 = 2.0, c2 = 2.0 -> c1*r1 = 1.0, c2*r2 = 1.0
    # v(t+1) = v(t) + 1.0 * (pbest - x) + 1.0 * (informant - x)

    # Particle 0:
    # v = [0.5, 0.5] + 1.0*([1,1] - [0,0]) + 1.0*([2,2] - [0,0])
    # v = [0.5, 0.5] + [1.0, 1.0] + [2.0, 2.0] = [3.5, 3.5]

    # Particle 1:
    # v = [0.0, 0.0] + 1.0*([1,1] - [1,1]) + 1.0*([2,2] - [1,1])
    # v = [0.0, 0.0] + [0.0, 0.0] + [1.0, 1.0] = [1.0, 1.0]

    # Particle 2:
    # v = [-0.5, -0.5] + 1.0*([-2,-2] - [-1,-1]) + 1.0*([2,2] - [-1,-1])
    # v = [-0.5, -0.5] + [-1.0, -1.0] + [3.0, 3.0] = [1.5, 1.5]

    expected_velocities = np.array([[3.5, 3.5], [1.0, 1.0], [1.5, 1.5]])

    np.testing.assert_array_equal(new_velocities, expected_velocities)
