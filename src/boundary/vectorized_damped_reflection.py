"""Vectorized SPSO 2011 damped reflection boundary handling."""

from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .vectorized_base import VectorizedBoundaryHandler


class VectorizedDampedReflectionBoundaryHandler(VectorizedBoundaryHandler):
    """
    SPSO 2011 boundary handler applied to the whole swarm at once.

    Positions that leave the search space are clamped to the violated bound and
    the corresponding velocity components are reflected and damped. Unlike
    position clamping, which zeroes the offending velocity, reflection keeps some
    momentum pointing back into the space.
    """

    def __init__(self, damping_factor: float = -0.5) -> None:
        """
        Args:
            damping_factor: Multiplier applied to velocity components that hit a
                bound. The SPSO 2011 default of -0.5 both reverses and halves them.
        """
        self._damping_factor = float(damping_factor)

    def apply(self, swarm_state: SwarmState, bounds: np.ndarray | None, **_kwargs: Any) -> None:
        """
        Clamp positions and reflect the violating velocity components in place.

        Args:
            swarm_state: The swarm to correct; ``positions`` and ``velocities``
                are modified in place.
            bounds: Array of shape (D, 2) holding per-dimension min and max, or
                None to disable boundary handling.
            **_kwargs: Unused; present for protocol compatibility.
        """
        if bounds is None:
            return

        min_bounds = bounds[:, 0]
        max_bounds = bounds[:, 1]

        # Strictly outside only: a particle resting exactly on a bound is legal
        # and must not have its velocity reversed.
        violated = (swarm_state.positions < min_bounds) | (swarm_state.positions > max_bounds)

        np.clip(swarm_state.positions, min_bounds, max_bounds, out=swarm_state.positions)
        swarm_state.velocities[violated] *= self._damping_factor
