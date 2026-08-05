"""Global Topology implementation for PSO neighborhood.

In a global topology, each particle is connected to all other particles in the swarm,
allowing information to spread quickly throughout the entire swarm.
"""

from ..particles.base import ParticleBase
from .base import NeighborhoodTopology


class GlobalTopology(NeighborhoodTopology):
    """
    Global topology implementation where each particle is connected to all other particles.
    In the global topology (also known as fully connected or star topology),
    each particle is influenced by the best solution found by any particle in the entire swarm.
    This typically leads to faster convergence but may increase the risk of premature convergence
    to local optima since diversity in the swarm can be quickly lost.
    """

    def get_neighbors(
        self, particles: list[ParticleBase], rebuild: bool | None = None
    ) -> dict[ParticleBase, list[ParticleBase]]:
        """
        Determine the neighbors for each particle in the swarm using global topology.
        In global topology, each particle's neighborhood consists of all other particles in the swarm.
        Args:
            particles: A list of all particles in the swarm.
        Returns:
            A dictionary mapping each particle to all other particles in the swarm.
        """
        _ = rebuild
        neighborhood = {}
        for particle in particles:
            # Each particle is connected to all other particles (excluding itself)
            neighborhood[particle] = [p for p in particles if p is not particle]
        return neighborhood
