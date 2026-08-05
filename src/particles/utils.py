"""Utility functions for simplified particle management.

This module provides direct particle creation utilities that eliminate
the need for factory pattern overhead when only one particle type exists.
"""

import numpy as np

from ..initialization.base import InitializationStrategy
from .base import ParticleBase
from .standard import StandardParticle


def create_swarm(initialization_strategy: InitializationStrategy, num_particles: int) -> list[ParticleBase]:
    """
    Create a swarm of particles directly without factory overhead.

    Args:
        initialization_strategy: Strategy for particle initialization
        num_particles: Number of particles to create

    Returns:
        List of initialized StandardParticle instances
    """
    positions, velocities = initialization_strategy.initialize_swarm(num_particles)
    return [StandardParticle(pos, vel) for pos, vel in zip(positions, velocities, strict=False)]


def create_particle(initialization_strategy: InitializationStrategy) -> StandardParticle:
    """
    Create a single particle directly without factory overhead.

    Args:
        initialization_strategy: Strategy for particle initialization

    Returns:
        Initialized StandardParticle instance
    """
    position, velocity = initialization_strategy.generate_next_particle()
    return StandardParticle(position, velocity)


def restart_particle(
    initialization_strategy: InitializationStrategy,
    override_bounds: np.ndarray | None = None,
    current_best_position: np.ndarray | None = None,
) -> StandardParticle:
    """
    Create a restarted particle directly without factory overhead.

    Args:
        initialization_strategy: Strategy for particle initialization
        override_bounds: Optional bounds override for restart
        current_best_position: Current global best position

    Returns:
        Newly initialized StandardParticle instance
    """
    position, velocity = initialization_strategy.generate_restart_state(
        override_bounds=override_bounds, _current_best_position=current_best_position
    )
    return StandardParticle(position, velocity)
