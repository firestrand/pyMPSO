import numpy as np

from ..clamping.position import VelocityResetPositionClampingStrategy, apply_position_clamping
from ..particles.base import ParticleBase
from .base import BoundaryHandler

"""Clamping boundary handling strategy implementation."""


class PositionClampingBoundaryHandler(BoundaryHandler):
    """
    Position clamping boundary handling strategy.
    This strategy:
    1. Clamps particle positions to the specified bounds
    2. Zeroes velocity components that would take the particle outside the bounds
    3. Preserves the direction of velocity for components that remain in bounds
    This is a simple and commonly used boundary handling strategy that ensures
    particles stay within the search space while maintaining their momentum
    in valid dimensions.
    """

    def __init__(self, reset_velocity: bool = True):
        """
        Initialize the clamping boundary handler.
        Args:
            reset_velocity: Whether to reset velocity components in dimensions where
                           position was clamped. Default is True.
        """
        self.reset_velocity = reset_velocity
        self.position_strategy = VelocityResetPositionClampingStrategy()

    def apply(self, particle: ParticleBase, bounds: np.ndarray) -> None:
        """
        Apply clamping boundary handling to a particle.
        Args:
            particle: The particle to handle boundaries for
            bounds: Array of shape (n_dimensions, 2) containing the lower and upper
                   bounds for each dimension
        """
        apply_position_clamping(
            particle=particle,
            bounds=bounds,
            position_strategy=self.position_strategy,
            reset_velocity=self.reset_velocity,
        )
