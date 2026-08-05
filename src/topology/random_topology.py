import numpy as np

from ..particles.base import ParticleBase
from .base import NeighborhoodTopology

"""Random neighborhood topology implementation for SPSO 2011."""


class RandomTopology(NeighborhoodTopology):
    """
    Random neighborhood topology implementation for SPSO 2011.
    In this topology, each particle is connected to K randomly selected
    informants (neighbors), which can change at each iteration. Each particle
    is always included in its own neighborhood.
    This implementation follows the SPSO 2011 standard where each particle
    has approximately K=3 informants on average.
    """

    def __init__(
        self,
        k: int = 3,
        seed: int | None = None,
        rebuild_probability: float = 1.0,
        rng: np.random.RandomState | None = None,
    ):
        """
        Initialize the random topology.
        Args:
            k: The approximate number of informants each particle should have.
               Default is 3 as per SPSO 2011 specification.
            seed: Optional random seed for reproducibility.
            rebuild_probability: Probability of rebuilding the neighborhood
                                topology at each call. Default is 1.0 (always rebuild).
            rng: Optional shared random generator to use for topology sampling.
        """
        self._k = k
        self._seed = seed
        self._rebuild_probability = rebuild_probability
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()
        self._cached_neighbors: dict[ParticleBase, list[ParticleBase]] | None = None

    def get_neighbors(
        self, particles: list[ParticleBase], rebuild: bool | None = None
    ) -> dict[ParticleBase, list[ParticleBase]]:
        """
        Determine the neighbors for each particle in the swarm.
        For each particle, approximately K random particles are selected as informants.
        Each particle is always included in its own neighborhood.
        Args:
            particles: A list of all particles in the swarm.
            rebuild: Optional override for topology rebuild behavior:
                - None (default): use stochastic rebuild_probability.
                - True: force rebuild.
                - False: reuse cached topology when available.
        Returns:
            A dictionary mapping each particle to its list of neighboring particles.
        """
        # Use cached topology if available.
        if self._cached_neighbors is not None:
            if rebuild is False:
                return self._cached_neighbors
            if rebuild is None and self._rng.random() > self._rebuild_probability:
                return self._cached_neighbors
        # Rebuild when requested or when rebuild is not specified and cached topology
        # is not eligible to be reused.
        # Create a new neighborhood structure
        neighbors = {}
        # The informant probability is based on the swarm size and desired connectivity K.
        # This is equivalent to the C SPSO2011 convention where edges are
        # sampled for each source particle (m) to each recipient particle (s), then
        # applied as informants for the recipient.
        n = len(particles)
        if n <= 1:
            if n == 1:
                neighbors[particles[0]] = [particles[0]]
            return neighbors
        # SPSO 2011 probability that a given particle selects another as an informant.
        # p = 1 - (1 - 1 / S)^K, where S is swarm size.
        # This keeps self always in neighborhood and gives expected degree
        # 1 + (S - 1) * p.
        inform_prob = 1 - ((n - 1) / n) ** self._k
        for s, particle_s in enumerate(particles):
            # Each recipient keeps itself as an informant.
            neighbors[particle_s] = [particle_s]
            # For each possible source informant m, draw m -> s.
            for m, particle_m in enumerate(particles):
                if m == s:
                    continue
                if self._rng.random() < inform_prob:
                    neighbors[particle_s].append(particle_m)
        # Cache the neighborhoods for potential reuse
        self._cached_neighbors = neighbors
        return neighbors
