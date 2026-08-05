from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..particles.swarm_state import SwarmState


@runtime_checkable
class PositionUpdateStrategy(Protocol):
    """Protocol for updating swarm positions based on velocities or other mechanics."""

    def update_positions(self, swarm_state: SwarmState, **kwargs: Any) -> np.ndarray:
        """
        Calculates and returns the new positions for the swarm.

        Args:
            swarm_state: The current vectorized state of the swarm.
            **kwargs: Additional parameters specific to the update strategy.

        Returns:
            np.ndarray: The updated positions matrix of shape (N, D).
        """
        ...
