import numpy as np

from ..particles.base import ParticleBase
from .base import NeighborhoodInfluenceStrategy

"""Single Best Influence Strategy implementation for PSO."""


class SingleBestInfluence(NeighborhoodInfluenceStrategy):
    """
    Influence strategy where a particle is influenced by the single best neighbor.
    This strategy identifies the particle with the best (lowest) pbest_fitness
    within the defined neighborhood (the particle itself and its actual neighbors)
    and uses its personal best position as the social influence component in the
    velocity update equation.
    """

    def get_informant_position(
        self, particle: ParticleBase, neighbors: list[ParticleBase], hyperparams: dict | None = None
    ) -> np.ndarray:
        """
        Get the personal best position of the single best particle in the neighborhood
        (particle + neighbors).
        Args:
            particle: The particle being updated.
            neighbors: List of neighboring particles that may influence this particle.
        Returns:
            The personal best position of the particle (including itself)
            with the best pbest_fitness in the neighborhood.
        """
        # Combine the particle itself with its neighbors to form the full neighborhood for influence
        # In C SPSO, ties for equal fitness keep the current particle as the local best.
        best_informant = particle
        best_fitness = particle.pbest_fitness
        for neighbor in neighbors:
            if neighbor.pbest_fitness < best_fitness:
                best_informant = neighbor
                best_fitness = neighbor.pbest_fitness

        if hyperparams is not None and hyperparams.get("consume_random_for_self_informant", False):
            rng = hyperparams.get("rng")
            if best_informant is particle and rng is not None:
                swarm_size = int(hyperparams.get("swarm_size", len(neighbors)))
                if swarm_size > 1:
                    particle_index = int(hyperparams.get("particle_index", -1))
                    while True:
                        candidate = int(np.floor(rng.random() * swarm_size))
                        if candidate != particle_index:
                            break

        return np.asarray(best_informant.pbest)
