from abc import ABC, abstractmethod

from ..particles.base import ParticleBase

"""Abstract Base Class for Neighborhood Topology in PSO."""


class NeighborhoodTopology(ABC):
    """
    Abstract Base Class for all neighborhood topology implementations.
    The neighborhood topology defines how particles are connected to each other
    and which particles influence each other during the optimization process.
    Different topologies can significantly impact the convergence behavior,
    exploration / exploitation balance, and overall performance of PSO.
    """

    @abstractmethod
    def get_neighbors(
        self, particles: list[ParticleBase], rebuild: bool | None = None
    ) -> dict[ParticleBase, list[ParticleBase]]:
        """
        Determine the neighbors for each particle in the swarm.
        Args:
            particles: A list of all particles in the swarm.
            rebuild: Optional override to force rebuild (`True`) or reuse cached
                neighbors (`False`) when supported by the topology.
        Returns:
            A dictionary mapping each particle to its list of neighboring particles.
            The keys are particle instances, and the values are lists of particle instances.
        """
        pass
