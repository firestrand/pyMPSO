"""Ring Topology implementation for PSO neighborhood as used in SPSO 2007.

In a ring topology, particles are arranged in a ring structure where each particle
is connected to k neighbors on each side, forming a neighborhood of 2k + 1 particles
(including itself).
"""

from ..particles.base import ParticleBase
from .base import NeighborhoodTopology


class RingTopology(NeighborhoodTopology):
    """
    Ring topology implementation where each particle is connected to k neighbors on each side.
    In the ring topology (also known as lbest topology), particles are arranged in a ring,
    and each particle is influenced by its k immediate neighbors on each side, plus itself.
    This typically leads to slower convergence compared to global topology but can
    maintain more diversity in the swarm and often results in better global solutions.
    SPSO 2007 typically uses k=1, meaning each particle is influenced by itself and
    its immediate left and right neighbors (a total of 3 informants).
    """

    def __init__(self, k: int = 1):
        """
        Initialize the ring topology with k neighbors on each side.
        Args:
            k: Number of neighbors on each side. Default is 1, resulting in
               a neighborhood influence size of 2k + 1 (particle + k neighbors on each side).
               For SPSO 2007, the standard setting is k=1.
        """
        if k < 1:
            raise ValueError("Number of neighbors (k) must be at least 1")
        self.k = k

    def get_neighbors(
        self, particles: list[ParticleBase], rebuild: bool | None = None
    ) -> dict[ParticleBase, list[ParticleBase]]:
        """
        Determine the neighbors for each particle in the swarm using ring topology.
        Args:
            particles: A list of all particles in the swarm.
        Returns:
            A dictionary mapping each particle to its neighbors in the ring structure.
        """
        _ = rebuild
        if not particles:
            return {}
        n = len(particles)
        if n <= 1:
            # If there's only one particle, it has no neighbors
            return {particles[0]: []}
        neighborhood = {}
        for i, particle in enumerate(particles):
            # Initialize empty list of neighbors
            neighbors = []
            # Add k neighbors on each side, wrapping around if needed
            for j in range(1, self.k + 1):
                # Left neighbor with wrap - around
                left_idx = (i - j) % n
                neighbors.append(particles[left_idx])
                # Right neighbor with wrap - around
                right_idx = (i + j) % n
                neighbors.append(particles[right_idx])
            neighborhood[particle] = neighbors
        return neighborhood
