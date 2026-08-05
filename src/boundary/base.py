from abc import ABC, abstractmethod

import numpy as np

from ..particles.base import ParticleBase

"""Abstract Base Class for Boundary Handling in PSO."""


class BoundaryHandler(ABC):
    """
    Abstract Base Class for all boundary handling strategies.
    The boundary handler defines how particles that move outside the valid
    search space are handled. Different strategies may have different effects
    on the optimization process and convergence.
    """

    @abstractmethod
    def apply(self, particle: ParticleBase, bounds: np.ndarray) -> None:
        """
        Apply boundary handling to a particle.
        This method modifies the particle's position and potentially velocity
        to ensure it respects the specified bounds.
        Args:
            particle: The particle to apply boundary handling to.
            bounds: A numpy array of shape (dimensions, 2) specifying the
                   lower and upper bounds for each dimension.
                   bounds[:, 0] are the lower bounds, bounds[:, 1] are the upper bounds.
        """
        pass
