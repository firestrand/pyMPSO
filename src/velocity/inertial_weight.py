import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy

"""Inertia Weight Velocity Update Strategy Implementation."""


class InertiaWeightVelocityUpdate(VelocityUpdateStrategy):
    """
    Inertia Weight Velocity Update Strategy.
    This is the most common and standard PSO velocity update formula, based on
    the following equation:
    v(t + 1) = w * v(t) + c1 * r1 * (p - x(t)) + c2 * r2 * (g - x(t))
    where:
    * w: inertia weight
    * c1: cognitive coefficient
    * c2: social coefficient
    * r1, r2: random values between 0 and 1
    * p: particle's personal best position
    * g: global best position (or neighborhood best)
    * x: current particle position
    The inertia weight (w) controls the impact of the previous velocity, with
    higher values favoring exploration and lower values favoring exploitation.
    Typically:
    * w ∈ [0.4, 0.9] (often 0.729 for optimized PSO variants)
    * c1, c2 ∈ [1.5, 2.5] (often both 1.49445 for optimized PSO variants)
    Inertia weight can be constant or varying over iterations (e.g., linearly
    decreasing) to balance exploration and exploitation.
    For boundary handling, this implementation supports velocity clamping where
    any coordinate that exceeds its bound is clamped to the bound and its velocity
    is adjusted accordingly.
    """

    def __init__(self, clamping_strategy: VelocityClampingStrategy | None = None):
        """
        Initialize the inertia weight velocity update strategy.
        Args:
            clamping_strategy: Strategy for velocity clamping. If None, NoClampingStrategy is used.
        """
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Update a particle's velocity using the inertia weight PSO formula.
        Args:
            particle: The particle whose velocity will be updated.
            informant_position: The position to use for the social component, typically
                                the global best or neighborhood best position.
            hyperparams: Dictionary containing the hyperparameters:
                * ``w`` - inertia weight (default 0.729)
                * ``c1`` - cognitive coefficient (default 1.49445)
                * ``c2`` - social coefficient (default 1.49445)
                * ``bounds`` – tuple / list of (min, max) arrays for clamping (optional)
        Returns:
            Updated (and possibly clamped) velocity vector.
        """
        # Extract parameters from hyperparams
        w = hyperparams.get("w", 0.729)
        c1 = hyperparams.get("c1", 1.49445)
        c2 = hyperparams.get("c2", 1.49445)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        # Random components
        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        # Calculate the cognitive component (attraction to personal best)
        cognitive = c1 * r1 * (particle.pbest - particle.position)
        # Calculate the social component (attraction to global / neighborhood best)
        social = c2 * r2 * (informant_position - particle.position)
        # Update velocity using the inertia weight formula
        v_new = w * particle.velocity + cognitive + social
        # Optional clamping ------------------------------------------------------
        if bounds is not None:
            clamped_velocity = self.clamping_strategy.clamp(v_new, bounds, hyperparams)
            # Ensure return type is ndarray
            v_new = np.asarray(clamped_velocity, dtype=np.float64)
        # Ensure we return a proper ndarray
        return np.asarray(v_new, dtype=np.float64)
