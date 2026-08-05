from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..particles.swarm_state import SwarmState


@runtime_checkable
class VectorizedBoundaryHandler(Protocol):
    """Protocol for vectorized boundary handlers."""

    def apply(self, swarm_state: SwarmState, bounds: np.ndarray, **kwargs: Any) -> None:
        """
        Applies boundary handling logic (e.g. clamping) to the entire swarm
        in-place or by modifying the SwarmState matrices.

        Args:
            swarm_state: The current vectorized state of the swarm.
            bounds: A numpy array of shape (D, 2) where column 0 is min,
                    and column 1 is max.
            **kwargs: Additional parameters.
        """
        ...
