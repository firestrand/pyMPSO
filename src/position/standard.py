from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .base import PositionUpdateStrategy


class StandardPositionUpdate(PositionUpdateStrategy):
    """
    Standard position update strategy for vectorized PSO.
    Calculates new positions as X(t+1) = X(t) + V(t+1).
    """

    def update_positions(self, swarm_state: SwarmState, **_kwargs: Any) -> np.ndarray:
        """
        Calculates and returns the new positions by adding the current velocities
        to the current positions.

        Args:
            swarm_state: The current vectorized state of the swarm.
            **kwargs: Additional parameters (unused in standard update).

        Returns:
            np.ndarray: The updated positions matrix of shape (N, D).
        """
        return swarm_state.positions + swarm_state.velocities
