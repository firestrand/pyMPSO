from abc import ABC, abstractmethod

import numpy as np

from ..particles.base import ParticleBase

"""Abstract Base Class for Velocity Update Strategy in PSO."""


class VelocityUpdateStrategy(ABC):
    """
    Abstract Base Class for all velocity update strategies.
    The velocity update strategy defines how a particle's velocity is updated
    during each iteration of the PSO algorithm. Different strategies can affect
    convergence behavior, exploration / exploitation balance, and solution quality.
    """

    @abstractmethod
    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Calculate the updated velocity for a particle.
        Args:
            particle: The particle being updated.
            informant_position: The position that influences the social component
                of the velocity update, typically derived from neighboring particles.
            hyperparams: Dictionary of hyperparameters used in the velocity update formula,
                which may include inertia weight, cognitive coefficient, social coefficient, etc.
        Returns:
            A numpy array representing the updated velocity.
        """
        pass
