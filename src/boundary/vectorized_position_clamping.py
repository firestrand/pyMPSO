from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .vectorized_base import VectorizedBoundaryHandler


class VectorizedPositionClampingBoundaryHandler(VectorizedBoundaryHandler):
    """
    Vectorized implementation of Position Clamping Boundary Handler.
    If a particle's position exceeds the bounds, its position is clamped
    to the bound, and its velocity in that dimension is set to zero.
    """

    def apply(self, swarm_state: SwarmState, bounds: np.ndarray | None, **_kwargs: Any) -> None:
        """
        Clamps swarm_state.positions in-place and zeros swarm_state.velocities
        where clamping occurred.
        """
        if bounds is None:
            return

        min_bounds = bounds[:, 0]
        max_bounds = bounds[:, 1]

        # Broadcasting min_bounds and max_bounds from shape (D,) to (N, D)
        # Find elements that are out of bounds
        exceeds_max = swarm_state.positions > max_bounds
        exceeds_min = swarm_state.positions < min_bounds

        # We need a boolean mask of any clamping
        clamped = exceeds_max | exceeds_min

        # Clamp positions in-place using np.clip
        # np.clip works seamlessly with broadcasted arrays
        np.clip(swarm_state.positions, min_bounds, max_bounds, out=swarm_state.positions)

        # Zero velocities where positions were clamped
        swarm_state.velocities[clamped] = 0.0
