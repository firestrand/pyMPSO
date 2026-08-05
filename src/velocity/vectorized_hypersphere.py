"""Vectorized SPSO 2011 hypersphere velocity update."""

from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import HypersphereVelocityConfig
from .vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedHypersphereVelocity(VectorizedVelocityUpdateStrategy):
    """
    Rotation-invariant SPSO 2011 velocity update over the whole swarm.

    For each particle a gravity centre ``G`` is derived from its position, its
    personal best and its informant, a point is drawn uniformly from the
    hypersphere of radius ``|G - x|`` centred on ``G``, and the new velocity is
    ``w*v + (sample - x)``.

    Being rotation invariant is the point of this update: the random draw uses a
    normalized Gaussian direction rather than per-dimension noise, so the search
    is not biased toward the coordinate axes.
    """

    def _gravity_centers(
        self,
        positions: np.ndarray,
        pbest_positions: np.ndarray,
        informant_matrix: np.ndarray,
        c1: float,
        c2: float,
    ) -> np.ndarray:
        """
        Compute the per-particle gravity centre.

        SPSO 2011 collapses the three-term average to two terms when a particle
        is its own informant, so that a lone best particle is not pulled toward
        a duplicate of its own personal best.

        Args:
            positions: Current positions, shape (N, D).
            pbest_positions: Personal best positions, shape (N, D).
            informant_matrix: Informant positions, shape (N, D).
            c1: Cognitive coefficient.
            c2: Social coefficient.

        Returns:
            The gravity centres, shape (N, D).
        """
        cognitive = pbest_positions - positions
        social = informant_matrix - positions

        three_term = positions + (c1 / 3.0) * cognitive + (c2 / 3.0) * social
        two_term = positions + 0.5 * c1 * cognitive

        informant_is_pbest = np.all(pbest_positions == informant_matrix, axis=1)
        return np.where(informant_is_pbest[:, None], two_term, three_term)

    def update_velocities(
        self,
        swarm_state: SwarmState,
        informant_matrix: np.ndarray,
        config: HypersphereVelocityConfig,
        **_kwargs: Any,
    ) -> np.ndarray:
        """
        Update velocities for the entire swarm using the hypersphere method.

        Args:
            swarm_state: The current vectorized state of the swarm.
            informant_matrix: Informant positions, shape (N, D).
            config: Supplies ``c1``, ``c2``, ``w``, ``distrib`` and ``rng``.
            **_kwargs: Unused; present for protocol compatibility.

        Returns:
            The updated velocity matrix, shape (N, D).
        """
        positions = swarm_state.positions
        dimensions = swarm_state.dimensions
        rng = getattr(config, "rng", None) or np.random

        gravity = self._gravity_centers(
            positions,
            swarm_state.pbest_positions,
            informant_matrix,
            float(config.c1),
            float(config.c2),
        )
        radii = np.linalg.norm(gravity - positions, axis=1)

        directions = rng.standard_normal(positions.shape)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # A zero draw has no direction; leaving it at zero keeps the sample at G.
        directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 0)

        u = rng.random(swarm_state.num_particles)
        radius_scale = u if int(config.distrib) == 0 else u ** (1.0 / dimensions)
        sampled = gravity + (radii * radius_scale)[:, None] * directions

        # A zero radius makes `sampled` equal to the position, so this reduces to
        # w*v without needing a separate branch.
        return float(config.w) * swarm_state.velocities + (sampled - positions)
