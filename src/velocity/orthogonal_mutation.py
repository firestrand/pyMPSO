"""Orthogonal mutation velocity update strategy.

This strategy extends the standard PSO velocity update with an orthogonal
mutation component. It is inspired by recent "orthogonally initiated" PSO
variants and is useful for injecting controlled directional diversity when
search progress stalls.
"""

from __future__ import annotations

import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..constants import STANDARD_C1, STANDARD_C2
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy


class OrthogonalMutationVelocityUpdate(VelocityUpdateStrategy):
    """Orthogonal-mutation PSO velocity update.

    The update is:
    v = standard_velocity + mutation_vector

    where the mutation vector is generated from an orthogonal basis and
    gated by a mutation probability.
    """

    def __init__(
        self,
        clamping_strategy: VelocityClampingStrategy | None = None,
        mutation_probability: float = 0.05,
        mutation_strength: float = 0.05,
        mutation_axes: int | None = None,
    ) -> None:
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()
        self.mutation_probability = float(mutation_probability)
        self.mutation_strength = float(mutation_strength)
        self.mutation_axes = mutation_axes

    def _sample_orthogonal_mutation(
        self,
        particle: ParticleBase,
        bounds: np.ndarray | None,
        mutation_axes: int,
        mutation_strength: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        dimensions = particle.position.shape[0]
        if dimensions <= 1 or mutation_axes <= 0:
            return np.zeros_like(particle.position)

        # Build an orthogonal basis from a random matrix and draw a random
        # combination of the first `mutation_axes` basis vectors.
        random_matrix = rng.standard_normal((dimensions, mutation_axes))
        orthogonal_basis, _ = np.linalg.qr(random_matrix)

        coeffs = rng.standard_normal(size=mutation_axes)
        raw_mutation = orthogonal_basis @ coeffs

        scale = np.ones(dimensions)
        if bounds is not None and bounds.size:
            span = np.abs(bounds[:, 1] - bounds[:, 0])
            # Keep a minimum scale for unbounded-ish or degenerate dimensions.
            span = np.where(span > 0, span, 1.0)
            scale = span

        return raw_mutation * mutation_strength * scale

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """Update velocity with optional orthogonal mutation."""

        c1 = hyperparams.get("c1", STANDARD_C1)
        c2 = hyperparams.get("c2", STANDARD_C2)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        mutation_probability = hyperparams.get("mutation_probability", self.mutation_probability)
        mutation_strength = hyperparams.get("mutation_strength", self.mutation_strength)
        mutation_axes = hyperparams.get("mutation_axes", self.mutation_axes)

        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        cognitive = c1 * r1 * (particle.pbest - particle.position)
        social = c2 * r2 * (informant_position - particle.position)
        velocity = particle.velocity + cognitive + social

        if rng.random() < mutation_probability:
            axes = int(mutation_axes or max(1, min(3, velocity.size)))
            axes = max(1, min(axes, velocity.size))
            mutation = self._sample_orthogonal_mutation(
                particle=particle,
                bounds=bounds,
                mutation_axes=axes,
                mutation_strength=float(mutation_strength),
                rng=rng,
            )
            velocity = velocity + mutation

        if bounds is not None:
            velocity = np.asarray(self.clamping_strategy.clamp(velocity, bounds, hyperparams), dtype=np.float64)

        return np.asarray(velocity, dtype=np.float64)
