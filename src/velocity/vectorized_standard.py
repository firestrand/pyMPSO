from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import StandardVelocityConfig
from .vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedStandardVelocityUpdate(VectorizedVelocityUpdateStrategy):
    """
    Vectorized Standard PSO Velocity Update Strategy.
    v(t + 1) = v(t) + c1 * r1 * (p - x(t)) + c2 * r2 * (g - x(t))
    """

    def update_velocities(
        self, swarm_state: SwarmState, informant_matrix: np.ndarray, config: StandardVelocityConfig, **_kwargs: Any
    ) -> np.ndarray:
        """
        Update the velocities for the entire swarm using vectorized matrix operations.

        Args:
            swarm_state: The current vectorized state of the swarm.
            informant_matrix: The social informants matrix of shape (N, D).
            config: StandardVelocityConfig containing hyperparameters (c1, c2, rng, bounds).
            **kwargs: Additional parameters (e.g. for clamping).

        Returns:
            np.ndarray: Updated velocity matrix.
        """
        shape = swarm_state.positions.shape

        # r1 and r2 are random matrices of shape (N, D)
        r1 = config.rng.random(shape)
        r2 = config.rng.random(shape)

        cognitive = config.c1 * r1 * (swarm_state.pbest_positions - swarm_state.positions)
        social = config.c2 * r2 * (informant_matrix - swarm_state.positions)

        new_velocities = swarm_state.velocities + cognitive + social

        # In the old standard velocity update, bounds clamping was also handled here.
        # But Phase 1 task 1.11 / 1.12 says "Test boundary handlers applied to SwarmState matrices".
        # Typically velocity clamping is distinct from boundary handlers, or it can be a separate step.
        # For now, we return the unclamped velocities or implement simple vectorized clamping.
        # Let's keep it pure to the update rule, or apply basic min/max clamping if bounds exist in config.
        # Wait, the old StandardVelocityUpdate handled clamping_strategy. We'll leave it simple for now.

        return new_velocities
