"""Vectorized random neighborhood topology for SPSO 2011."""

from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import RandomTopologyConfig
from .vectorized_base import VectorizedTopologyStrategy


class VectorizedRandomTopology(VectorizedTopologyStrategy):
    """
    SPSO 2011 random topology built as a single adjacency matrix.

    Every particle informs itself. Each other directed link is drawn
    independently with probability ``p = 1 - ((S-1)/S)^K``, which gives the SPSO
    2011 expected degree of ``1 + (S-1)*p``. The whole matrix is sampled in one
    draw rather than looping over particle pairs.

    The strategy is stateful: the sampled matrix is cached and reused according
    to ``rebuild_probability``, matching the scalar ``RandomTopology``. Note that
    the random draws happen in a single batch, so the RNG stream differs from the
    scalar implementation's nested loop; parity with the C reference is validated
    statistically, not draw-by-draw.
    """

    def __init__(self) -> None:
        self._cached_matrix: np.ndarray | None = None

    def get_topology_matrix(
        self,
        swarm_state: SwarmState,
        config: RandomTopologyConfig,
        **_kwargs: Any,
    ) -> np.ndarray:
        """
        Return the ``(N, N)`` boolean adjacency matrix, resampling it when due.

        Args:
            swarm_state: The current vectorized state of the swarm.
            config: Random topology configuration supplying ``k``,
                ``rebuild_probability`` and the run ``rng``.
            **_kwargs: Unused; present for protocol compatibility.

        Returns:
            A boolean matrix where entry ``(i, j)`` is True when particle ``j``
            informs particle ``i``. The diagonal is always True.
        """
        n = swarm_state.num_particles
        rng = getattr(config, "rng", None) or np.random.RandomState()
        rebuild_probability = float(getattr(config, "rebuild_probability", 1.0))

        cached = self._cached_matrix
        # A resized swarm invalidates the cache regardless of rebuild policy.
        if cached is not None and cached.shape == (n, n) and rng.random() > rebuild_probability:
            return cached

        k = int(getattr(config, "k", 3))
        if n <= 1:
            matrix = np.ones((n, n), dtype=bool)
        else:
            # SPSO 2011: probability that a given particle informs another.
            inform_prob = 1 - ((n - 1) / n) ** k
            matrix = rng.random((n, n)) < inform_prob
            np.fill_diagonal(matrix, True)

        self._cached_matrix = matrix
        return matrix
