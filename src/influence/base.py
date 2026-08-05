from abc import ABC, abstractmethod

import numpy as np

from ..particles.base import ParticleBase

"""Abstract Base Class for Neighborhood Influence Strategy in PSO."""


class NeighborhoodInfluenceStrategy(ABC):
    """
    Abstract Base Class for all neighborhood influence strategies.
    The neighborhood influence strategy defines how information from neighboring particles
    is used to influence the velocity update of a particle. Different influence strategies
    can affect the convergence behavior and exploration / exploitation balance of PSO.
    """

    @abstractmethod
    def get_informant_position(
        self, particle: ParticleBase, neighbors: list[ParticleBase], hyperparams: dict | None = None
    ) -> np.ndarray:
        """
        Determine the informant position that influences a particle's velocity update.
        Args:
            particle: The particle being updated.
            neighbors: List of neighboring particles that may influence this particle.
            hyperparams: Optional algorithm context used by the influence strategy.
        Returns:
            A numpy array representing the informant position to be used in the velocity update.
        """
        pass
