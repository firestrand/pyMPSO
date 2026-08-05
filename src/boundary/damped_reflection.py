import numpy as np

from ..particles.base import ParticleBase
from .base import BoundaryHandler

"""Damped reflection boundary handling for SPSO 2011."""


class DampedReflectionBoundaryHandler(BoundaryHandler):
    """
    SPSO 2011 boundary handler that clamps particles to bounds and bounces velocity.
    This boundary handler:
    1. Clamps the particle position to the boundary when it exceeds it
    2. Reflects the velocity and dampens it by a factor of -0.5 (standard in SPSO 2011)
    This implementation follows the Standard PSO 2011 specification.
    """

    def __init__(self, bounds: np.ndarray | None = None, damping_factor: float = -0.5):
        """
        Initialize the damped reflection boundary handler.
        Args:
            bounds: A numpy array of shape (dimensions, 2) where each row
                   contains [min_bound, max_bound] for a dimension.
            damping_factor: Factor to multiply the velocity component when reflecting.
                          Default is -0.5 as specified in SPSO 2011.
        """
        super().__init__()
        self._bounds = bounds
        self._damping_factor = damping_factor

    def enforce_bounds(
        self, position: np.ndarray, velocity: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Enforce bounds on the particle position and apply velocity reflection if needed.
        When a position component exceeds the boundary:
        1. The position is clamped to the boundary
        2. The corresponding velocity component is reflected and scaled by damping_factor
        Args:
            position: Current position of the particle
            velocity: Current velocity of the particle (optional)
        Returns:
            A tuple containing:
            - The adjusted position (clamped to bounds if needed)
            - The adjusted velocity (reflected and damped if needed, or None if not provided)
        """
        # Check if bounds are available
        if self._bounds is None:
            return position, velocity
        # Get lower and upper bounds for each dimension
        lower_bounds = self._bounds[:, 0]
        upper_bounds = self._bounds[:, 1]
        # Create masks for positions outside bounds
        below_min = position < lower_bounds
        above_max = position > upper_bounds
        # Create a copy of the position to modify
        adjusted_position = position.copy()
        # Clamp positions to boundaries
        adjusted_position[below_min] = lower_bounds[below_min]
        adjusted_position[above_max] = upper_bounds[above_max]
        # If velocity is provided, adjust it for components that are outside bounds
        adjusted_velocity = None
        if velocity is not None:
            adjusted_velocity = velocity.copy()
            # Apply damped reflection to velocity components
            # For components that hit the boundary, flip and scale the velocity
            adjusted_velocity[below_min] *= self._damping_factor
            adjusted_velocity[above_max] *= self._damping_factor
        return adjusted_position, adjusted_velocity

    def apply(self, particle: ParticleBase, bounds: np.ndarray) -> None:
        """
        Apply boundary handling to a particle.
        This method enforces the bounds by clamping the particle's position
        and applying damped reflection to its velocity when needed.
        Args:
            particle: The particle to apply boundary handling to.
            bounds: A numpy array of shape (dimensions, 2) specifying the
                   lower and upper bounds for each dimension.
        """
        # Store the bounds for use in enforce_bounds
        self._bounds = bounds
        # Call the enforce_bounds method to handle boundary violations
        adjusted_position, adjusted_velocity = self.enforce_bounds(particle.position, particle.velocity)
        # Update the particle's position and velocity
        particle.position = adjusted_position
        if adjusted_velocity is not None:
            particle.velocity = adjusted_velocity
