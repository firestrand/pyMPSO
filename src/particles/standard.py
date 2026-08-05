import numpy as np

from .base import ParticleBase

"""Standard Particle implementation for PSO."""

# import sys # No longer needed


class StandardParticle(ParticleBase):
    """
    Represents a standard particle in the Particle Swarm Optimization algorithm.
    Implements the ParticleBase interface with core position, velocity, and
    personal best tracking.
    """

    def __init__(self, initial_position: np.ndarray, initial_velocity: np.ndarray):
        """
        Initializes a StandardParticle with a given position and velocity.
        Args:
            initial_position: The starting position numpy array.
            initial_velocity: The starting velocity numpy array.
        Raises:
            ValueError: If initial_position or initial_velocity are not valid numpy arrays
                        of the same 1D shape.
        """
        if not isinstance(initial_position, np.ndarray) or initial_position.ndim != 1:
            raise ValueError("initial_position must be a 1D numpy array.")
        if not isinstance(initial_velocity, np.ndarray) or initial_velocity.shape != initial_position.shape:
            raise ValueError("initial_velocity must be a 1D numpy array with the same shape as initial_position.")
        self._dimensions = initial_position.shape[0]
        # Initialize directly from arguments
        self._position: np.ndarray = initial_position.copy()  # Use copy to avoid external mutation
        self._velocity: np.ndarray = initial_velocity.copy()  # Use copy
        # Initialize pbest based on initial state
        self._pbest: np.ndarray = initial_position.copy()
        self._pbest_fitness: float = float(np.finfo(np.float64).max)  # Assume initial fitness unknown / high
        self._fitness: float = float(np.finfo(np.float64).max)  # Needs evaluation
        # Initialize tracking variables
        self._iterations_since_improvement: int = 0
        self._update_count: int = 0
        # Initialize metadata dictionary for algorithm - specific data
        self._metadata: dict = {}

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) or value.shape != (self._dimensions,):
            raise ValueError(f"Position must be a numpy array of shape ({self._dimensions},)")
        # Only increment update_count if the position actually changes
        if not np.array_equal(self._position, value):
            self._position = value
            self._update_count += 1
        else:
            self._position = value

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) or value.shape != (self._dimensions,):
            raise ValueError(f"Velocity must be a numpy array of shape ({self._dimensions},)")
        self._velocity = value

    @property
    def pbest(self) -> np.ndarray:
        return self._pbest

    @pbest.setter
    def pbest(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) or value.shape != (self._dimensions,):
            raise ValueError(f"Personal best position must be a numpy array of shape ({self._dimensions},)")
        self._pbest = value

    @property
    def pbest_fitness(self) -> float:
        return self._pbest_fitness

    @pbest_fitness.setter
    def pbest_fitness(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("Personal best fitness must be a float or int.")
        self._pbest_fitness = float(value)

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("Fitness must be a float or int.")
        self._fitness = float(value)

    @property
    def iterations_since_improvement(self) -> int:
        return self._iterations_since_improvement

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def metadata(self) -> dict:
        return self._metadata

    def update_pbest(self) -> None:
        """
        Updates the particle's personal best position and fitness
        if the current position yields a better fitness.
        Assumes lower fitness values are better.
        """
        # Check if current position is better than personal best
        if self._fitness < self._pbest_fitness:
            self._pbest = self._position.copy()
            self._pbest_fitness = self._fitness
            self._iterations_since_improvement = 0
        else:
            # Increment stagnation counter if no improvement
            self._iterations_since_improvement += 1

    def __repr__(self) -> str:
        return (
            f"StandardParticle(dim={self._dimensions}, pos={self._position}, vel={self._velocity}, "
            f"fitness={self._fitness:.4f}, "
            f"pbest_pos={self._pbest}, pbest_fit={self._pbest_fitness:.4f}, "
            f"age={self._update_count}, stagnation={self._iterations_since_improvement})"
        )
