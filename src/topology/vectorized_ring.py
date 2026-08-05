"""Vectorized ring (lbest) topology as used in SPSO 2007."""

from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .config import RingTopologyConfig
from .vectorized_base import VectorizedTopologyStrategy


class VectorizedRingTopology(VectorizedTopologyStrategy):
    """
    Ring topology over the whole swarm as a single adjacency matrix.

    Particles are arranged in a ring and each one is informed by its ``k`` nearest
    neighbours on either side plus itself, giving a neighbourhood of ``2k + 1``.
    The adjacency is built from the circular index distance, so the whole matrix
    is produced without iterating over particles.
    """

    def get_topology_matrix(
        self,
        swarm_state: SwarmState,
        config: RingTopologyConfig,
        **_kwargs: Any,
    ) -> np.ndarray:
        """
        Build the ``(N, N)`` boolean adjacency matrix for the ring.

        Args:
            swarm_state: The current vectorized state of the swarm.
            config: Ring configuration supplying ``k``, the number of neighbours
                on each side.
            **_kwargs: Unused; present for protocol compatibility.

        Returns:
            A boolean matrix where entry ``(i, j)`` is True when particle ``j``
            informs particle ``i``. The diagonal is always True.

        Raises:
            ValueError: If ``k`` is less than 1.
        """
        k = int(getattr(config, "k", 1))
        if k < 1:
            raise ValueError("Number of neighbors (k) must be at least 1")

        n = swarm_state.num_particles
        indices = np.arange(n)
        # Circular distance between every pair of ring positions.
        offset = np.abs(indices[:, None] - indices[None, :])
        circular_distance = np.minimum(offset, n - offset)
        return circular_distance <= k
