import numpy as np

from .base import VelocityClampingStrategy

"""Velocity clamping strategies implementation."""


class MaxNormVelocityClampingStrategy(VelocityClampingStrategy):
    """
    Clamps the velocity to a maximum absolute value per dimension.
    This strategy enforces a maximum velocity in each dimension by clamping
    each component to the range [-max_velocity, max_velocity].
    """

    def __init__(self, max_velocity: float | None = None):
        """
        Initialize the max norm velocity clamping strategy.
        Args:
            max_velocity: Optional fixed maximum velocity value. If None,
                          will use 'max_velocity' from hyperparams during clamp.
        """
        self.max_velocity = max_velocity

    def clamp(self, values: np.ndarray, bounds: np.ndarray, hyperparams: dict) -> np.ndarray:  # noqa: ARG002
        """
        Clamp velocity components to a maximum absolute value.
        Args:
            values: The velocity vector to clamp.
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters, can contain 'max_velocity'.
        Returns:
            The clamped velocity vector.
        """
        # Use instance value if set, otherwise check hyperparams
        max_velocity = self.max_velocity or hyperparams.get("max_velocity")
        if max_velocity is None:
            return np.asarray(values)  # No clamping if max_velocity not specified
        return np.asarray(np.clip(values, -max_velocity, max_velocity))


class RelativeBoundsVelocityClampingStrategy(VelocityClampingStrategy):
    """
    Clamps velocity based on a proportion of the search space dimensions.
    This strategy calculates the maximum velocity as a fraction of the search
    space range for each dimension, ensuring particles can't move more than
    a certain percentage of the search space in a single step.
    """

    def __init__(self, max_velocity_ratio: float = 0.1):
        """
        Initialize with max velocity as a proportion of search space range.
        Args:
            max_velocity_ratio: Maximum velocity as a proportion of bounds range.
                Default is 0.1 (10% of range per iteration).
        """
        self.max_velocity_ratio = max_velocity_ratio

    def clamp(self, values: np.ndarray, bounds: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Clamp velocity components based on search space dimensions.
        Args:
            values: The velocity vector to clamp.
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters (ratio can be overridden with 'max_velocity_ratio').
        Returns:
            The clamped velocity vector.
        """
        # Use ratio from hyperparams if provided, otherwise use instance ratio
        ratio = hyperparams.get("max_velocity_ratio", self.max_velocity_ratio)
        # Calculate range for each dimension
        ranges = bounds[:, 1] - bounds[:, 0]
        # Calculate maximum velocity for each dimension
        max_velocities = ranges * ratio
        # Clamp each dimension
        clamped_velocity = np.copy(values)
        for i in range(len(values)):
            clamped_velocity[i] = np.clip(values[i], -max_velocities[i], max_velocities[i])
        return np.asarray(clamped_velocity)


class NoClampingStrategy(VelocityClampingStrategy):
    """
    A strategy that applies no clamping to velocity.
    This is a pass - through strategy that can be used as a default.
    """

    def clamp(self, values: np.ndarray, bounds: np.ndarray, hyperparams: dict) -> np.ndarray:  # noqa: ARG002
        """
        Apply no clamping to velocity.
        Args:
            values: The velocity vector.
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters (not used).
        Returns:
            The original velocity vector, unchanged.
        """
        return values
