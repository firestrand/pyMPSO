import numpy as np

from src.influence.vectorized_single_best import VectorizedSingleBestInfluence
from src.particles.swarm_state import SwarmState


def test_vectorized_single_best_influence():
    """Verify SingleBestInfluence finds the correct informant per particle."""
    n_particles = 4
    dimensions = 2

    # pbest_positions
    # P0: [0, 0] (fit 10)
    # P1: [1, 1] (fit 5)
    # P2: [2, 2] (fit 8)
    # P3: [3, 3] (fit 2)

    pbest_positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    pbest_fitness = np.array([10.0, 5.0, 8.0, 2.0])

    state = SwarmState(
        positions=np.zeros((n_particles, dimensions)),
        velocities=np.zeros((n_particles, dimensions)),
        pbest_positions=pbest_positions,
        pbest_fitness=pbest_fitness,
        current_fitness=np.zeros(n_particles),
    )

    # Topology mask:
    # P0 sees P0, P1
    # P1 sees P1, P2
    # P2 sees P2, P3
    # P3 sees P0, P3
    topology_matrix = np.array(
        [[True, True, False, False], [False, True, True, False], [False, False, True, True], [True, False, False, True]]
    )

    strategy = VectorizedSingleBestInfluence()
    informant_matrix = strategy.get_informant_matrix(state, topology_matrix)

    # Expected informants:
    # P0 sees P0(10), P1(5) -> P1 is best -> [1, 1]
    # P1 sees P1(5), P2(8) -> P1 is best -> [1, 1]
    # P2 sees P2(8), P3(2) -> P3 is best -> [3, 3]
    # P3 sees P0(10), P3(2) -> P3 is best -> [3, 3]

    expected = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]])

    np.testing.assert_array_equal(informant_matrix, expected)


def test_vectorized_single_best_influence_global():
    """Verify SingleBestInfluence finds the global best for all particles if fully connected."""
    n_particles = 3
    dimensions = 2

    pbest_positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],  # best
            [2.0, 2.0],
        ]
    )
    pbest_fitness = np.array([10.0, 1.0, 8.0])

    state = SwarmState(
        positions=np.zeros((n_particles, dimensions)),
        velocities=np.zeros((n_particles, dimensions)),
        pbest_positions=pbest_positions,
        pbest_fitness=pbest_fitness,
        current_fitness=np.zeros(n_particles),
    )

    # Global topology
    topology_matrix = np.ones((n_particles, n_particles), dtype=bool)

    strategy = VectorizedSingleBestInfluence()
    informant_matrix = strategy.get_informant_matrix(state, topology_matrix)

    # Everyone should get P1
    expected = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    np.testing.assert_array_equal(informant_matrix, expected)
