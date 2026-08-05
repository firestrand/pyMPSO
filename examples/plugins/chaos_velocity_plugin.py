"""Example plugin component for research prototypes.

This module is intentionally tiny and self-contained so researchers can copy and
extend it directly when adding new velocity update rules.
"""

from __future__ import annotations

import numpy as np

from src.components.factory import register_component
from src.particles.base import ParticleBase
from src.velocity.base import VelocityUpdateStrategy


@register_component("velocity_strategy", "chaos_velocity")
class ChaosVelocityUpdate(VelocityUpdateStrategy):
    """Simple chaotic velocity perturbation strategy."""

    def __init__(self, chaos_scale: float = 0.1):
        self.chaos_scale = float(chaos_scale)

    def update(
        self,
        particle: ParticleBase,
        informant_position: np.ndarray,
        hyperparams: dict,
    ) -> np.ndarray:
        """Update velocity using chaotic perturbation plus standard attraction terms."""

        c1 = float(hyperparams.get("c1", 1.5))
        c2 = float(hyperparams.get("c2", 1.5))
        rng = hyperparams.get("rng", np.random)

        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        cognitive = c1 * r1 * (particle.pbest - particle.position)
        social = c2 * r2 * (informant_position - particle.position)
        chaos = rng.uniform(-self.chaos_scale, self.chaos_scale, size=particle.position.shape)

        return particle.velocity + cognitive + social + chaos
