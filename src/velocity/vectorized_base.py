from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..particles.swarm_state import SwarmState


@runtime_checkable
class VectorizedVelocityUpdateStrategy(Protocol):
    """Protocol for vectorized velocity update strategies."""

    def update_velocities(
        self, swarm_state: SwarmState, informant_matrix: np.ndarray, config: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Calculates and returns the new velocities for the entire swarm.

        Args:
            swarm_state: The current vectorized state of the swarm.
            informant_matrix: A matrix of shape (N, D) representing the informant
                              position for each particle.
            config: A strongly-typed configuration object for the strategy.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: The updated velocity matrix of shape (N, D).
        """
        ...
