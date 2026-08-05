import numpy as np

from ..particles.base import ParticleBase
from .base import PositionClampingStrategy

"""Position clamping strategies implementation."""


class BasicPositionClampingStrategy(PositionClampingStrategy):
    """
    Basic position clamping strategy that simply clips positions to bounds.
    This strategy ensures particles stay within the search space by clamping
    any out - of - bounds position coordinates to the nearest boundary value.
    It does not modify the particle's velocity.
    """

    def clamp(
        self,
        values: np.ndarray,
        bounds: np.ndarray,
        hyperparams: dict,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Clamp position values to stay within bounds.
        Args:
            values: The position vector to clamp.
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters (not used for this strategy).
        Returns:
            The clamped position vector.
        """
        return np.asarray(np.clip(values, bounds[:, 0], bounds[:, 1]))


class VelocityResetPositionClampingStrategy:
    """
    Position clamping helper that also provides velocity reset information.
    This helper:
    1. Clamps particle positions to the specified bounds
    2. Returns a tuple (clamped_position, mask) where the mask indicates dimensions
       that were clamped and may need velocity component resets
    Note: This is not a PositionClampingStrategy subclass because it returns additional
    information (the reset mask) beyond just the clamped values.
    """

    def clamp(
        self,
        values: np.ndarray,
        bounds: np.ndarray,
        hyperparams: dict,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Clamp position values and return a mask for velocity reset.
        Args:
            values: The position vector to clamp.
            bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds for each dimension.
            hyperparams: Dictionary of hyperparameters (not used for this strategy).
        Returns:
            A tuple containing:
            - The clamped position vector
            - Boolean mask indicating dimensions where clamping occurred (True where clamped)
        """
        # Find dimensions where position would be outside bounds
        below_lower = values < bounds[:, 0]
        above_upper = values > bounds[:, 1]
        # Create mask for dimensions that need velocity reset
        reset_mask = below_lower | above_upper
        # Clamp positions to bounds
        clamped_position = np.clip(values, bounds[:, 0], bounds[:, 1])
        return clamped_position, reset_mask


# Helper function for applying position clamping to particles


def apply_position_clamping(
    particle: ParticleBase,
    bounds: np.ndarray,
    position_strategy: PositionClampingStrategy | VelocityResetPositionClampingStrategy,
    reset_velocity: bool = False,
    hyperparams: dict | None = None,
) -> None:
    """
    Apply position clamping to a particle with optional velocity reset.
    This helper function applies a position clamping strategy to a particle and
    optionally resets velocity components in dimensions where clamping occurred.
    Args:
        particle: The particle to apply clamping to.
        bounds: Array of shape (n_dimensions, 2) containing [min, max] bounds.
        position_strategy: The position clamping strategy to use.
        reset_velocity: Whether to reset velocity components in clamped dimensions.
        hyperparams: Optional hyperparameters dictionary to pass to the clamping strategy.
    """
    if hyperparams is None:
        hyperparams = {}
    # Basic position clamping
    if isinstance(position_strategy, BasicPositionClampingStrategy) or not reset_velocity:
        result = position_strategy.clamp(particle.position, bounds, hyperparams)
        if isinstance(result, tuple):
            particle.position = result[0]
        else:
            particle.position = np.asarray(result)
    # Position clamping with velocity reset
    elif isinstance(position_strategy, VelocityResetPositionClampingStrategy):
        clamped_position, reset_mask = position_strategy.clamp(particle.position, bounds, hyperparams)
        particle.position = clamped_position
        # Reset velocity components in dimensions where position was clamped
        if reset_velocity:
            particle.velocity[reset_mask] = 0.0
    # Generic fallback for other position clamping strategies
    else:
        result = position_strategy.clamp(particle.position, bounds, hyperparams)
        if isinstance(result, tuple) and len(result) == 2:
            clamped_position, reset_mask = result
            particle.position = clamped_position
            # Reset velocity components in dimensions where position was clamped
            if reset_velocity:
                particle.velocity[reset_mask] = 0.0
        else:
            particle.position = np.asarray(result)
