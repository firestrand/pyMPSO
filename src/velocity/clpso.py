"""Comprehensive Learning PSO (CLPSO) Velocity Update Strategy.

Implementation of CLPSO by Liang et al. (2006):
"Comprehensive learning particle swarm optimizer for global optimization of multimodal functions"

Key Innovation: Each particle learns from different exemplars for different dimensions,
preventing premature convergence and improving performance on multimodal problems.

Reference:
Liang, J. J., Qin, A. K., Suganthan, P. N., & Baskar, S. (2006).
Comprehensive learning particle swarm optimizer for global optimization of multimodal functions.
IEEE transactions on evolutionary computation, 10(3), 281 - 295.
"""

from typing import Any

import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy


class CLPSOVelocityUpdate(VelocityUpdateStrategy):
    """
    Comprehensive Learning PSO Velocity Update Strategy.
    In CLPSO, each dimension of a particle can learn from different exemplars'
    personal best positions. The learning probability Pc determines whether
    a particle should update its exemplar list.
    Key features:
    - Each dimension can have a different exemplar
    - Learning probability Pc controls exemplar refresh rate
    - Inertia weight decreases linearly over iterations
    - No global best is used - particles learn from each other's pbests
    Velocity Update Formula:
    v_i(t + 1) = w * v_i(t) + c * r * (exemplar_i - x_i(t))
    where exemplar_i is the personal best of the selected exemplar for dimension i.
    """

    def __init__(
        self,
        clamping_strategy: VelocityClampingStrategy | None = None,
        c: float = 1.49445,
        w_start: float = 0.9,
        w_end: float = 0.4,
        refresh_gap: int = 7,
    ):
        """
        Initialize the CLPSO velocity update strategy.
        Args:
            clamping_strategy: Strategy for velocity clamping. If None, NoClampingStrategy is used.
            c: Acceleration coefficient (typically 1.49445).
            w_start: Initial inertia weight (typically 0.9).
            w_end: Final inertia weight (typically 0.4).
            refresh_gap: Number of iterations without improvement before refreshing exemplars.
        """
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()
        self.c = c
        self.w_start = w_start
        self.w_end = w_end
        self.refresh_gap = refresh_gap
        # Storage for exemplar assignments (particle -> dimension -> exemplar index)
        self._exemplar_assignments: dict[int, np.ndarray] = {}
        # Storage for stagnation counters (particle -> count)
        self._stagnation_counters: dict[int, int] = {}
        # Storage for previous best fitness (particle -> fitness)
        self._previous_best_fitness: dict[int, float] = {}

    def _calculate_learning_probability(self, particle_index: int, swarm_size: int) -> float:
        """
        Calculate the learning probability Pc for a particle.
        The learning probability is calculated based on particle index to ensure diversity.
        Better particles (lower index after sorting by fitness) have lower Pc.
        Args:
            particle_index: Index of the particle in the swarm (0 - based).
            swarm_size: Total number of particles in the swarm.
        Returns:
            Learning probability Pc in range [0.05, 0.5].
        """
        if swarm_size <= 1:
            return 0.5
        # Linear interpolation from 0.5 to 0.05 based on particle rank
        pc = 0.5 - 0.45 * (particle_index / (swarm_size - 1))
        return float(np.clip(pc, 0.05, 0.5))

    def _generate_exemplar_assignments(
        self,
        particle: ParticleBase,
        all_particles: list[ParticleBase],
        pc: float,
        rng: Any,
    ) -> np.ndarray:
        """
        Generate exemplar assignments for each dimension of a particle.
        For each dimension, select an exemplar particle to learn from based on
        tournament selection among random particles.
        Args:
            particle: The particle to generate exemplars for.
            all_particles: All particles in the swarm.
            pc: Learning probability for this particle.
        Returns:
            Array of exemplar indices for each dimension.
        """
        num_particles = len(all_particles)
        num_dimensions = len(particle.position)
        exemplars = np.zeros(num_dimensions, dtype=int)
        # Get particle index
        particle_idx = all_particles.index(particle)
        for d in range(num_dimensions):
            if rng.random() < pc:
                # Learn from another particle's pbest
                # Tournament selection between two random particles
                if num_particles >= 2:
                    candidates = rng.choice(num_particles, size=2, replace=False)
                    # Choose the better one (lower pbest_fitness)
                    if all_particles[candidates[0]].pbest_fitness < all_particles[candidates[1]].pbest_fitness:
                        exemplars[d] = candidates[0]
                    else:
                        exemplars[d] = candidates[1]
                else:
                    # Only one particle - use self
                    exemplars[d] = particle_idx
            else:
                # Learn from own pbest
                exemplars[d] = particle_idx
        return np.asarray(exemplars, dtype=int)

    def _should_refresh_exemplars(self, particle: ParticleBase) -> bool:
        """
        Check if a particle should refresh its exemplar assignments.
        Refresh occurs when the particle has not improved for refresh_gap iterations.
        Args:
            particle: The particle to check.
        Returns:
            True if exemplars should be refreshed, False otherwise.
        """
        particle_id = id(particle)
        # Initialize if first time
        if particle_id not in self._stagnation_counters:
            self._stagnation_counters[particle_id] = 0
            self._previous_best_fitness[particle_id] = particle.pbest_fitness
            return True  # Generate initial exemplars
        # Check for improvement
        if particle.pbest_fitness < self._previous_best_fitness[particle_id]:
            # Improved - reset counter
            self._stagnation_counters[particle_id] = 0
            self._previous_best_fitness[particle_id] = particle.pbest_fitness
            return False
        else:
            # No improvement - increment counter
            self._stagnation_counters[particle_id] += 1
            # Check if refresh needed
            if self._stagnation_counters[particle_id] >= self.refresh_gap:
                self._stagnation_counters[particle_id] = 0
                return True
        return False

    def update(
        self,
        particle: ParticleBase,
        informant_position: np.ndarray,  # noqa: ARG002
        hyperparams: dict,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Update a particle's velocity using the CLPSO strategy.
        Args:
            particle: The particle whose velocity will be updated.
            informant_position: Kept for interface consistency with other velocity update strategies.
            hyperparams: Dictionary containing the hyperparameters:
                * 'all_particles' - List of all particles in the swarm (required)
                * 'current_iteration' - Current iteration number (optional)
                * 'max_iterations' - Maximum iterations (optional)
                * 'bounds' - tuple / list of (min, max) arrays for clamping (optional)
        Returns:
            Updated (and possibly clamped) velocity vector.
        Raises:
            ValueError: If 'all_particles' is not provided in hyperparams.
        """
        # Extract required parameters
        _ = informant_position
        all_particles = hyperparams.get("all_particles")
        if all_particles is None:
            raise ValueError("CLPSO requires 'all_particles' in hyperparams")
        bounds = hyperparams.get("bounds")
        current_iteration = hyperparams.get("current_iteration", 0)
        max_iterations = hyperparams.get("max_iterations", 1000)
        rng = hyperparams.get("rng", np.random)
        particle_id = id(particle)
        num_particles = len(all_particles)
        # Calculate particle index (for learning probability)
        try:
            particle_idx = all_particles.index(particle)
        except ValueError:
            particle_idx = 0  # Default if particle not in list
        # Calculate learning probability
        pc = self._calculate_learning_probability(particle_idx, num_particles)
        # Check if exemplars need refresh
        if self._should_refresh_exemplars(particle) or particle_id not in self._exemplar_assignments:
            self._exemplar_assignments[particle_id] = self._generate_exemplar_assignments(
                particle,
                all_particles,
                pc,
                rng,
            )
        # Get current exemplar assignments
        exemplars = self._exemplar_assignments[particle_id]
        # Calculate adaptive inertia weight
        w = self.w_start - (self.w_start - self.w_end) * (current_iteration / max_iterations)
        # Update velocity dimension by dimension
        new_velocity = np.zeros_like(particle.velocity)
        for d in range(len(particle.position)):
            # Get exemplar's pbest for this dimension
            exemplar_particle = all_particles[exemplars[d]]
            exemplar_position = exemplar_particle.pbest[d]
            # Generate random coefficient
            r = rng.random()
            # Update velocity for this dimension
            new_velocity[d] = w * particle.velocity[d] + self.c * r * (exemplar_position - particle.position[d])
        # Apply velocity clamping strategy
        if bounds is not None:
            clamped_velocity = self.clamping_strategy.clamp(new_velocity, bounds, hyperparams)
            new_velocity = np.asarray(clamped_velocity, dtype=np.float64)
        # Ensure we return a proper ndarray
        return np.asarray(new_velocity, dtype=np.float64)

    def get_required_hyperparams(self) -> list[str]:
        """
        Get the list of required hyperparameters for CLPSO.
        Returns:
            List of required hyperparameter names.
        """
        return ["all_particles"]

    def get_optional_hyperparams(self) -> dict[str, Any]:
        """
        Get the dictionary of optional hyperparameters with their default values.
        Returns:
            Dictionary of optional hyperparameter names and default values.
        """
        return {"bounds": None, "current_iteration": 0, "max_iterations": 1000}

    def reset(self):
        """Reset internal state for a new optimization run."""
        self._exemplar_assignments.clear()
        self._stagnation_counters.clear()
        self._previous_best_fitness.clear()

    def __str__(self) -> str:
        """String representation of the CLPSO velocity strategy."""
        return f"CLPSOVelocityUpdate(c={self.c:.3f}, w=[{self.w_start:.1f},{self.w_end:.1f}], gap={self.refresh_gap})"

    def __repr__(self) -> str:
        """Detailed string representation of the CLPSO velocity strategy."""
        return (
            f"CLPSOVelocityUpdate(c={self.c}, w_start={self.w_start}, "
            f"w_end={self.w_end}, refresh_gap={self.refresh_gap}, "
            f"clamping_strategy={type(self.clamping_strategy).__name__})"
        )
