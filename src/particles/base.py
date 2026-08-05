from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

"""Abstract Base Class for Particle representations in PSO."""


class ParticleBase(ABC):
    """
    Abstract Base Class for all particle types.
    Defines the essential properties and methods required for a particle
    in the PSO algorithm.
    """

    @property
    @abstractmethod
    def position(self) -> NDArray[np.float64]:
        """Current position of the particle in the search space."""
        pass

    @position.setter
    @abstractmethod
    def position(self, value: NDArray[np.float64]) -> None:
        pass

    @property
    @abstractmethod
    def velocity(self) -> NDArray[np.float64]:
        """Current velocity of the particle."""
        pass

    @velocity.setter
    @abstractmethod
    def velocity(self, value: NDArray[np.float64]) -> None:
        pass

    @property
    @abstractmethod
    def pbest(self) -> NDArray[np.float64]:
        """Personal best position found by the particle so far."""
        pass

    @pbest.setter
    @abstractmethod
    def pbest(self, value: NDArray[np.float64]) -> None:
        pass

    @property
    @abstractmethod
    def pbest_fitness(self) -> float:
        """Fitness value of the personal best position."""
        pass

    @pbest_fitness.setter
    @abstractmethod
    def pbest_fitness(self, value: float) -> None:
        pass

    @property
    @abstractmethod
    def fitness(self) -> float:
        """Current fitness value of the particle."""
        pass

    @fitness.setter
    @abstractmethod
    def fitness(self, value: float) -> None:
        pass

    @property
    @abstractmethod
    def iterations_since_improvement(self) -> int:
        """Number of iterations since the particle's last fitness improvement."""
        pass

    @property
    @abstractmethod
    def update_count(self) -> int:
        """
        Number of times the particle's position has been updated.
        This effectively tracks the particle's age in iterations.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """
        Additional metadata for the particle that can be used by various
        PSO variants and strategies.
        This can include algorithm - specific data like:
        - Acceleration coefficients for adaptive PSO
        - Historical trajectory information
        - Diversity metrics
        - Neighborhood information
        """
        pass

    @abstractmethod
    def update_pbest(self) -> None:
        """
        Updates the particle's personal best position and fitness
        if the current position is better.
        """
        pass
