import numpy as np

from .base import InitializationStrategy

"""Random uniform initialization strategy for particle states."""

# No longer need ParticleBase or StandardParticle here
# from particles.base import ParticleBase
# from particles.standard import StandardParticle


class RandomUniformInitialization(InitializationStrategy):
    """
    Initialization strategy that generates particle states uniformly at random
    within the search space bounds.
    """

    def generate_next_particle(self, override_bounds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the next particle state by sampling uniformly at random from the bounds.
        Args:
            override_bounds: Optional bounds to use for this specific particle generation.
        Returns:
            A tuple of (position, velocity) arrays for the particle.
        """
        # Use override bounds if provided, otherwise use default bounds
        bounds = override_bounds if override_bounds is not None else self._bounds
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        # Generate position uniformly within bounds
        position = self._rng.uniform(lower_bounds, upper_bounds)
        # Initialize velocity uniformly within [-Vmax, Vmax] where Vmax = Xmax - Xmin
        v_max = upper_bounds - lower_bounds
        velocity = self._rng.uniform(-v_max, v_max)
        return position, velocity

    def _should_use_local_search(self) -> bool:
        """
        Protected method to determine if local search should be used for restart.
        This method can be overridden in tests to control the random decision.
        Default implementation uses 30% probability.
        Returns:
            True if local search should be used, False otherwise.
        """
        return self._rng.random() < 0.3

    def generate_restart_state(
        self,
        override_bounds: np.ndarray | None = None,
        _current_best_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a restarted particle state with optional focus around the current best position.
        This maintains consistency with previously generated particles while allowing
        optional focusing around the current best position.
        Args:
            override_bounds: Optional bounds to use for this specific restart.
            current_best_position: Current global best position that may guide the restart.
        Returns:
            A tuple of (position, velocity) arrays for the restarted particle.
        """
        # Use the bounds provided or fall back to the default
        if override_bounds is not None:
            # Create a temporary strategy with different bounds but same state
            temp_strategy = self.with_bounds_override(override_bounds)
            # Use the temporary strategy's next particle generation
            pos, vel = temp_strategy.generate_next_particle()
            return pos, vel
        # Initialize with zero velocity
        velocity = np.zeros(self._dimensions)
        lower_bounds = self._bounds[:, 0]
        upper_bounds = self._bounds[:, 1]
        # If we have current best position and decide to use it (30% probability)
        if _current_best_position is not None and self._should_use_local_search():
            # Create a localized search around the best position
            # Use a range of 20% of the total dimension range
            ranges = upper_bounds - lower_bounds
            local_ranges = ranges * 0.2
            # Make sure the position stays within bounds
            local_lower = np.maximum(_current_best_position - local_ranges / 2, lower_bounds)
            local_upper = np.minimum(_current_best_position + local_ranges / 2, upper_bounds)
            # Generate the position in this local area
            position = self._rng.uniform(local_lower, local_upper, size=self._dimensions)
        else:
            # Get the next particle from the sequence
            position, _ = self.generate_next_particle()
        return position, velocity
