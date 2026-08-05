from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .vectorized_base import VectorizedTopologyStrategy


class VectorizedGlobalTopology(VectorizedTopologyStrategy):
    """
    Global topology implementation for vectorized PSO.
    Each particle is connected to all other particles in the swarm.
    """

    def get_topology_matrix(
        self,
        swarm_state: SwarmState,
        config: Any,  # noqa: ARG002
        **_kwargs: Any,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Returns a boolean adjacency matrix where all entries are True,
        meaning every particle is in the neighborhood of every other particle.
        """
        N = swarm_state.num_particles
        return np.ones((N, N), dtype=bool)
