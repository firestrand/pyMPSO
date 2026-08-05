from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import AdaptivePSOVelocityConfig
from .vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedAdaptivePSOVelocityUpdate(VectorizedVelocityUpdateStrategy):
    """Vectorized Adaptive PSO (APSO) velocity update."""

    @staticmethod
    def _interpolate(start: float, end: float, ratio: float) -> float:
        return start + (end - start) * ratio

    def update_velocities(
        self,
        swarm_state: SwarmState,
        informant_matrix: np.ndarray,
        config: AdaptivePSOVelocityConfig,
        *,
        current_iteration: int = 0,
        max_iterations: int | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        if max_iterations is None or max_iterations <= 0:
            schedule_ratio = 0.0
        else:
            schedule_ratio = min(max(float(current_iteration) / float(max_iterations), 0.0), 1.0)

        inertia = self._interpolate(config.w_min, config.w_max, schedule_ratio)
        c1 = self._interpolate(config.c1_max, config.c1_min, schedule_ratio)
        c2 = self._interpolate(config.c2_min, config.c2_max, schedule_ratio)

        shape = swarm_state.positions.shape
        r1 = config.rng.random(shape)
        r2 = config.rng.random(shape)

        cognitive = c1 * r1 * (swarm_state.pbest_positions - swarm_state.positions)
        social = c2 * r2 * (informant_matrix - swarm_state.positions)

        return inertia * swarm_state.velocities + cognitive + social
