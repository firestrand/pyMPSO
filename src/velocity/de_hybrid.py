"""Differential Evolution-inspired PSO velocity update strategy."""

from __future__ import annotations

import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..constants import STANDARD_C1, STANDARD_C2
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy


class DEHybridVelocityUpdate(VelocityUpdateStrategy):
    """Velocity update that injects DE-style perturbation into social guidance."""

    def __init__(
        self,
        clamping_strategy: VelocityClampingStrategy | None = None,
        differential_weight: float = 0.5,
        crossover_probability: float = 0.9,
    ) -> None:
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()
        self.differential_weight = float(differential_weight)
        self.crossover_probability = float(crossover_probability)

    @staticmethod
    def _random_distinct_indices(
        rng: np.random.RandomState, swarm_size: int, excluded_index: int
    ) -> tuple[int, int, int]:
        indices = [i for i in range(swarm_size) if i != excluded_index]
        chosen = rng.choice(indices, size=3, replace=False)
        return int(chosen[0]), int(chosen[1]), int(chosen[2])

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        c1 = hyperparams.get("c1", STANDARD_C1)
        c2 = hyperparams.get("c2", STANDARD_C2)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        differential_weight = hyperparams.get("differential_weight", self.differential_weight)
        crossover_probability = hyperparams.get("crossover_probability", self.crossover_probability)
        all_particles = hyperparams.get("all_particles")
        particle_index = int(hyperparams.get("particle_index", -1))

        # Base PSO pull toward informant.
        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        cognitive = c1 * r1 * (particle.pbest - particle.position)
        social = c2 * r2 * (informant_position - particle.position)
        velocity = particle.velocity + cognitive + social

        # Optional DE-derived diversity injection.
        if all_particles and len(all_particles) >= 4 and particle_index >= 0:
            r1_idx, r2_idx, r3_idx = self._random_distinct_indices(rng, len(all_particles), particle_index)
            a = all_particles[r1_idx].pbest
            b = all_particles[r2_idx].pbest
            c = all_particles[r3_idx].pbest
            mutant = a + differential_weight * (b - c)

            # Binomial crossover in decision space.
            dim = particle.position.shape[0]
            cross_mask = rng.random(dim) < crossover_probability
            if not cross_mask.any():
                cross_mask[rng.randint(0, dim)] = True
            trial = np.where(cross_mask, mutant, particle.pbest)

            # Nudge toward DE trial as mutation vector.
            velocity = velocity + 0.5 * (trial - particle.position)

        if bounds is not None:
            velocity = np.asarray(self.clamping_strategy.clamp(velocity, bounds, hyperparams), dtype=np.float64)

        return np.asarray(velocity, dtype=np.float64)
