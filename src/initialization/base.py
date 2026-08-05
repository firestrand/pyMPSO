from abc import ABC, abstractmethod

import numpy as np

"""Abstract Base Class for strategies that generate initial particle states."""

# No longer need ParticleBase here
# from particles.base import ParticleBase


class InitializationStrategy(ABC):
    """
    Abstract Base Class for strategies to generate initial states (positions, velocities)
    for a swarm of particles.
    This class maintains internal state to ensure consistent sequential generation
    of particle states, similar to a generator.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
    ):
        """
        Initialize the strategy with the bounds for the search space.
        Args:
            bounds: A numpy array of shape (dimensions, 2) with [min, max] bounds for each dimension.
            seed: Optional random seed for reproducible sequences.
            rng: Optional shared random generator to use for initialization.
        """
        # Validate bounds
        if not isinstance(bounds, np.ndarray) or len(bounds.shape) != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must be a numpy array of shape (dimensions, 2)")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Lower bounds cannot be greater than upper bounds")
        self._bounds = bounds
        self._dimensions = bounds.shape[0]
        self._seed = seed
        # Initialize random state for consistent sequence generation.
        # Keep an optional shared RNG for exact coupling with other algorithm components.
        if rng is not None:
            self._rng = rng
        elif seed is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(seed)
        # Track initialization progress
        self._initialized = False

    @property
    def dimensions(self) -> int:
        """Get the dimensionality of the search space defined by the bounds."""
        return int(self._dimensions)

    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds of the search space."""
        return self._bounds.copy()  # Return a copy to prevent external modification

    def initialize_swarm(self, num_particles: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate initial states for a swarm of particles.
        Args:
            num_particles: The number of particle states to generate.
        Returns:
            A tuple containing two lists:
            - List of initial position numpy arrays.
            - List of initial velocity numpy arrays.
        """
        self._initialized = True
        return self._generate_batch(num_particles)

    @abstractmethod
    def generate_next_particle(self, override_bounds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the next particle state in sequence, useful for particle restarts.
        This is the base method that concrete implementations must provide.
        It generates a single particle state, optionally using override bounds.
        Args:
            override_bounds: Optional bounds to use for this specific particle generation.
        Returns:
            A tuple of (position, velocity) arrays for the particle.
        """
        pass

    def _generate_batch(
        self, batch_size: int, override_bounds: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a batch of particle states by repeatedly calling generate_next_particle.
        Args:
            batch_size: Number of particle states to generate.
            override_bounds: Optional bounds to use for all particles in the batch.
        Returns:
            A tuple containing two lists:
            - List of position numpy arrays.
            - List of velocity numpy arrays.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        for _ in range(batch_size):
            position, velocity = self.generate_next_particle(override_bounds)
            positions.append(position)
            velocities.append(velocity)
        return positions, velocities

    def with_bounds_override(self, override_bounds: np.ndarray):
        """
        Create a new strategy instance with overridden bounds.
        This allows temporarily changing bounds while preserving the original strategy's state.
        Args:
            override_bounds: The new bounds to use.
        Returns:
            A new initialization strategy with the same state but different bounds.
        """
        # Validate the override bounds
        if not isinstance(override_bounds, np.ndarray) or override_bounds.shape != self._bounds.shape:
            raise ValueError(f"Override bounds must be a numpy array of shape {self._bounds.shape}")
        if np.any(override_bounds[:, 0] > override_bounds[:, 1]):
            raise ValueError("Lower bounds cannot be greater than upper bounds in override_bounds")
        # Create a new instance with the same seed but different bounds
        new_instance = self.__class__(bounds=override_bounds)
        # Copy the random state to ensure continued sequence.
        if hasattr(self._rng, "get_state") and hasattr(new_instance._rng, "set_state"):
            new_instance._rng.set_state(self._rng.get_state())
        new_instance._initialized = self._initialized
        return new_instance

    def generate_restart_state(
        self,
        override_bounds: np.ndarray | None = None,
        _current_best_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a single particle state for restarting a stagnated or reset particle.
        This default implementation calls generate_next_particle with optional bounds override,
        but derived classes can override this to implement specialized restart logic.
        Args:
            override_bounds: Optional bounds to use for this specific restart.
            _current_best_position: Optional current global best position that may inform the restart strategy.
        Returns:
            A tuple of (position, velocity) arrays for the restarted particle.
        """
        # By default, just use the same logic as for next particle
        # But subclasses can override this to provide specialized restart behavior
        return self.generate_next_particle(override_bounds)

    def _generate_states_with_optional_bounds(
        self, num_particles: int, override_bounds: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Internal helper method that temporarily changes bounds if override_bounds is provided.
        Args:
            num_particles: Number of particle states to generate.
            override_bounds: Optional bounds to use instead of the default bounds.
        Returns:
            A tuple containing two lists:
            - List of initial position numpy arrays.
            - List of initial velocity numpy arrays.
        """
        if override_bounds is not None:
            # Validate the override bounds
            if not isinstance(override_bounds, np.ndarray) or override_bounds.shape != self._bounds.shape:
                raise ValueError(f"Override bounds must be a numpy array of shape {self._bounds.shape}")
            if np.any(override_bounds[:, 0] > override_bounds[:, 1]):
                raise ValueError("Lower bounds cannot be greater than upper bounds in override_bounds")
            # Temporarily set override bounds
            original_bounds = self._bounds
            self._bounds = override_bounds
            try:
                # Generate states with temporary bounds
                return self._generate_batch(num_particles)
            finally:
                # Always restore original bounds
                self._bounds = original_bounds
        else:
            # Use normal bounds
            return self._generate_batch(num_particles)
