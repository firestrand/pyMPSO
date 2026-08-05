"""FIPS (Fully Informed Particle Swarm) Influence Strategy.

This influence strategy works with FIPSVelocityUpdate to provide
neighborhood information to particles. Unlike traditional influence
strategies that select the best informant, FIPS uses ALL neighbors.
"""

import numpy as np

from ..particles.base import ParticleBase
from .base import NeighborhoodInfluenceStrategy


class FIPSInfluenceStrategy(NeighborhoodInfluenceStrategy):
    """
    FIPS Influence Strategy that provides all neighbors to particles.
    This strategy doesn't select a single best informant like traditional
    strategies. Instead, it passes all neighbors to the velocity update
    strategy, which will use information from ALL neighbors.
    The get_informant_position method returns a dummy position since FIPS
    doesn't use a single informant. The actual neighbor information is
    passed through the enhanced hyperparameters.
    """

    def get_informant_position(
        self, particle: ParticleBase, neighbors: list[ParticleBase], hyperparams: dict | None = None
    ) -> np.ndarray:
        """
        Get the informant position for a particle.
        For FIPS, this method returns a dummy position since FIPS doesn't
        use a single informant. The actual logic is handled by storing
        neighbors in the particle's context for later use.
        Args:
            particle: The particle seeking an informant
            neighbors: List of neighboring particles (including potentially the particle itself)
        Returns:
            Dummy informant position (not used by FIPS velocity update)
        """
        _ = hyperparams

        if not neighbors:
            # If no neighbors, use particle's own personal best
            return np.asarray(particle.pbest.copy())
        # Return the global best as a dummy (FIPS velocity update will ignore this)
        # The actual neighbors are passed through hyperparameters in FIPS - aware algorithms
        best_neighbor = min(neighbors, key=lambda p: p.pbest_fitness)
        return np.asarray(best_neighbor.pbest.copy())

    def __str__(self) -> str:
        """String representation of the FIPS influence strategy."""
        return "FIPSInfluenceStrategy"

    def __repr__(self) -> str:
        """Detailed string representation of the FIPS influence strategy."""
        return "FIPSInfluenceStrategy()"
