from abc import ABC, abstractmethod

import numpy as np

"""Abstract Base Classes for Clamping Strategies in PSO."""


class ClampingStrategy(ABC):
    """
    Abstract Base Class for all value clamping strategies.
    This class defines the core interface for any strategy that restricts or bounds
    numerical values (like positions, velocities) within certain limits.
    """

    @abstractmethod
    def clamp(
        self, values: np.ndarray, bounds: np.ndarray, hyperparams: dict
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Apply clamping to a vector of values.
        Args:
            values: The values to clamp (could be position, velocity, etc.).
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters related to clamping.
        Returns:
            The clamped values as a numpy array.
        """
        pass


class PositionClampingStrategy(ClampingStrategy):
    """
    Abstract base class for position - specific clamping strategies.
    Position clamping typically involves ensuring particles stay within the search space
    boundaries, possibly with specific behaviors when boundaries are violated.
    """

    pass


class VelocityClampingStrategy(ClampingStrategy):
    """
    Abstract base class for velocity - specific clamping strategies.
    Velocity clamping typically involves limiting the magnitude of velocity
    components to prevent explosion and improve search behavior.
    """

    pass
