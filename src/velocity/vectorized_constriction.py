from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import ConstrictionVelocityConfig
from .vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedConstrictionVelocityUpdate(VectorizedVelocityUpdateStrategy):
    """
    Vectorized Constriction Coefficient PSO Velocity Update Strategy.
    v(t + 1) = χ * [v(t) + φ1 * r1 * (p - x(t)) + φ2 * r2 * (g - x(t))]
    """

    @staticmethod
    def _calculate_constriction_coefficient(phi1: float, phi2: float) -> float:
        phi = phi1 + phi2
        if phi <= 4:
            phi = 4.1  # Default fallback
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4 * phi))
        return float(chi)

    def update_velocities(
        self, swarm_state: SwarmState, informant_matrix: np.ndarray, config: ConstrictionVelocityConfig, **_kwargs: Any
    ) -> np.ndarray:

        chi = (
            config.chi if config.chi is not None else self._calculate_constriction_coefficient(config.phi1, config.phi2)
        )
        shape = swarm_state.positions.shape

        r1 = config.rng.random(shape)
        r2 = config.rng.random(shape)

        cognitive = config.phi1 * r1 * (swarm_state.pbest_positions - swarm_state.positions)
        social = config.phi2 * r2 * (informant_matrix - swarm_state.positions)

        new_velocities = chi * (swarm_state.velocities + cognitive + social)

        return new_velocities
