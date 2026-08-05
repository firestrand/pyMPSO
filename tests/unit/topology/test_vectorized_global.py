import numpy as np

from src.particles.swarm_state import SwarmState
from src.topology.config import GlobalTopologyConfig
from src.topology.vectorized_global import VectorizedGlobalTopology


def test_vectorized_global_topology():
    """Verify GlobalTopology correctly identifies all particles as neighbors."""
    n_particles = 4
    dimensions = 2

    state = SwarmState(
        positions=np.zeros((n_particles, dimensions)),
        velocities=np.zeros((n_particles, dimensions)),
        pbest_positions=np.zeros((n_particles, dimensions)),
        pbest_fitness=np.zeros(n_particles),
        current_fitness=np.zeros(n_particles),
    )

    strategy = VectorizedGlobalTopology()
    config = GlobalTopologyConfig()

    topology_matrix = strategy.get_topology_matrix(state, config)

    # Global topology should connect everyone to everyone.
    # Usually including self, or excluding self?
    # The old GlobalTopology excluded self: `neighborhood[particle] = [p for p in particles if p is not particle]`
    # Wait, the influence strategy `SingleBestInfluence` then combined the particle itself with its neighbors.
    # To keep things simple and pure, the topology matrix should just be True everywhere
    # except the diagonal (or including the diagonal, it doesn't really matter if influence adds self anyway).
    # Let's include self in the adjacency matrix so we don't have to specially handle it later.
    # Actually, let's see what is easier. If it's a mask, `np.ones((N, N), dtype=bool)` is including self.

    assert topology_matrix.shape == (4, 4)
    assert topology_matrix.dtype == bool
    # Let's say we just use True for all elements (including self)
    np.testing.assert_array_equal(topology_matrix, np.ones((4, 4), dtype=bool))
