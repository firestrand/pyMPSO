from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..particles.swarm_state import SwarmState


@runtime_checkable
class VectorizedInfluenceStrategy(Protocol):
    """Protocol for vectorized influence strategies."""

    def get_informant_matrix(self, swarm_state: SwarmState, topology_matrix: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Calculates and returns the informant matrix for the entire swarm.

        Args:
            swarm_state: The current vectorized state of the swarm.
            topology_matrix: A boolean matrix of shape (N, N) where (i, j) is True
                             if particle j influences particle i.
            **kwargs: Additional configuration parameters (e.g. rng).

        Returns:
            np.ndarray: An informant matrix of shape (N, D). The i-th row is the
                        informant position for the i-th particle.
        """
        ...
