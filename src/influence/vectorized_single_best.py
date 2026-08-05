from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .vectorized_base import VectorizedInfluenceStrategy


class VectorizedSingleBestInfluence(VectorizedInfluenceStrategy):
    """
    Vectorized implementation of Single Best Influence.
    Finds the particle with the lowest pbest_fitness in each neighborhood
    defined by the topology_matrix, and returns its pbest_position.
    """

    def get_informant_matrix(self, swarm_state: SwarmState, topology_matrix: np.ndarray, **_kwargs: Any) -> np.ndarray:
        """
        Calculates the informant matrix where row i is the pbest_position of the
        best neighbor of particle i.
        """

        # We need to find the min pbest_fitness for each row in the topology_matrix.
        # Create a masked fitness matrix where non-neighbors have infinity fitness.
        # np.where is fast: if it's a neighbor, use its fitness, else inf.

        # pbest_fitness is shape (N,). Broadcast to (N, N) so each row is a copy of pbest_fitness.
        # This means row i represents the fitness of all N particles.
        fitness_matrix = np.where(topology_matrix, swarm_state.pbest_fitness, np.inf)

        # Find the index of the minimum fitness in each row.
        # best_indices is an array of length N, where the i-th element is the
        # index of the best neighbor for particle i.
        best_indices = np.argmin(fitness_matrix, axis=1)

        # The informant matrix is just selecting these best_indices from pbest_positions.
        informant_matrix = swarm_state.pbest_positions[best_indices]

        # The original SingleBestInfluence has some tie-breaking and random-self-informant
        # logic for SPSO 2011 if the global best is the particle itself.
        # For a clean vectorization, we can handle the basic case first.
        # If 'consume_random_for_self_informant' is True, we could apply that logic here.
        # Let's keep it simple and clean. The minimum index returned by argmin is the
        # first one it encounters if there's a tie, which is consistent and deterministic.

        return informant_matrix
