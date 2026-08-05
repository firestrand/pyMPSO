from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import QuantumPSOVelocityConfig
from .vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedQPSOVelocityUpdate(VectorizedVelocityUpdateStrategy):
    """Vectorized Quantum-behaved PSO (QPSO) velocity update."""

    @staticmethod
    def _interpolate_beta(
        config: QuantumPSOVelocityConfig, current_iteration: int, max_iterations: int | None
    ) -> float:
        if config.beta_min is None or config.beta_max is None:
            return config.beta

        if max_iterations is None or max_iterations <= 0:
            schedule_ratio = 0.0
        else:
            schedule_ratio = min(max(float(current_iteration) / float(max_iterations), 0.0), 1.0)

        return config.beta_max + (config.beta_min - config.beta_max) * schedule_ratio

    def update_velocities(
        self,
        swarm_state: SwarmState,
        informant_matrix: np.ndarray,
        config: Any,
        **_kwargs: Any,
    ) -> np.ndarray:
        del informant_matrix
        if not isinstance(config, QuantumPSOVelocityConfig):
            raise TypeError("config must be an instance of QuantumPSOVelocityConfig")
        shape = swarm_state.positions.shape
        if shape[0] == 0:
            return np.empty_like(swarm_state.positions)

        current_iteration = int(_kwargs.get("current_iteration", 0))
        max_iterations = _kwargs.get("max_iterations")
        if max_iterations is not None:
            max_iterations = int(max_iterations)

        beta = self._interpolate_beta(config, current_iteration=current_iteration, max_iterations=max_iterations)

        mbest = np.mean(swarm_state.pbest_positions, axis=0, keepdims=True)
        attractor_weights = config.rng.random(shape)
        local_best = attractor_weights * swarm_state.pbest_positions + (1.0 - attractor_weights) * mbest

        rand_u = np.clip(config.rng.random(shape), 1e-12, 1.0)
        step_scale = beta * np.abs(local_best - swarm_state.positions) * np.log(1.0 / rand_u)
        random_sign = np.where(config.rng.random(shape) < 0.5, 1.0, -1.0)
        return random_sign * step_scale
