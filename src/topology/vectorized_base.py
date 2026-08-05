from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..particles.swarm_state import SwarmState


@runtime_checkable
class VectorizedTopologyStrategy(Protocol):
    """Protocol for vectorized topology strategies."""

    def get_topology_matrix(self, swarm_state: SwarmState, config: Any, **kwargs: Any) -> np.ndarray:
        """
        Calculates and returns the topology (neighborhood) matrix for the entire swarm.

        Args:
            swarm_state: The current vectorized state of the swarm.
            config: A strongly-typed configuration object for the topology.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: A boolean adjacency matrix of shape (N, N) where entry (i, j)
                        is True if particle j is in the neighborhood of particle i.
        """
        ...
