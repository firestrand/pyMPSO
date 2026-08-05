from dataclasses import dataclass

import numpy as np


@dataclass
class SwarmState:
    """
    Represents the vectorized state of the entire swarm.
    All attributes are NumPy arrays for high-performance matrix operations.
    """

    positions: np.ndarray
    velocities: np.ndarray
    pbest_positions: np.ndarray
    pbest_fitness: np.ndarray
    current_fitness: np.ndarray

    def __post_init__(self) -> None:
        """Validate the dimensions and shapes of the swarm state matrices."""
        if self.positions.ndim != 2:
            raise ValueError(f"positions must be a 2D array (num_particles, dimensions), got {self.positions.ndim}D")

        self.num_particles, self.dimensions = self.positions.shape

        if self.velocities.shape != (self.num_particles, self.dimensions):
            raise ValueError(
                f"velocities shape mismatch: expected {(self.num_particles, self.dimensions)}, "
                f"got {self.velocities.shape}"
            )

        if self.pbest_positions.shape != (self.num_particles, self.dimensions):
            raise ValueError(
                f"pbest_positions shape mismatch: expected {(self.num_particles, self.dimensions)}, "
                f"got {self.pbest_positions.shape}"
            )

        if self.pbest_fitness.shape != (self.num_particles,):
            raise ValueError(
                f"pbest_fitness shape mismatch: expected {(self.num_particles,)}, got {self.pbest_fitness.shape}"
            )

        if self.current_fitness.shape != (self.num_particles,):
            raise ValueError(
                f"current_fitness shape mismatch: expected {(self.num_particles,)}, got {self.current_fitness.shape}"
            )

    def get_global_best(self) -> tuple[np.ndarray, float]:
        """
        Return a copy of the current global best position and its fitness.
        This assumes pbest_fitness tracks the best-ever fitness per particle.
        """
        best_index = int(np.argmin(self.pbest_fitness))
        return self.pbest_positions[best_index].copy(), float(self.pbest_fitness[best_index])
