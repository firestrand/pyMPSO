import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..constants import STANDARD_C1, STANDARD_C2
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy

"""Standard Velocity Update Strategy Implementation."""


class StandardVelocityUpdate(VelocityUpdateStrategy):
    """
    Standard PSO Velocity Update Strategy.
    This is the classical PSO velocity update formula with separate cognitive and social
    coefficients:
    v(t + 1) = v(t) + c1 * r1 * (p - x(t)) + c2 * r2 * (g - x(t))
    where:
    * c1: cognitive coefficient
    * c2: social coefficient
    * r1, r2: random values between 0 and 1
    * p: particle's personal best position
    * g: global best position (or neighborhood best)
    * x: current particle position
    This standard form does not include an explicit inertia weight, but instead
    preserves the previous velocity directly.
    """

    def __init__(self, clamping_strategy: VelocityClampingStrategy | None = None):
        """
        Initialize the standard velocity update strategy.
        Args:
            clamping_strategy: Strategy for velocity clamping. If None, NoClampingStrategy is used.
        """
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Update a particle's velocity using the standard PSO formula.
        Args:
            particle: The particle whose velocity will be updated.
            informant_position: The position to use for the social component, typically
                                the global best or neighborhood best position.
            hyperparams: Dictionary containing the hyperparameters:
                * ``c1`` - cognitive coefficient (default 2.0)
                * ``c2`` - social coefficient (default 2.0)
                * ``bounds`` – tuple / list of (min, max) arrays for clamping (optional)
        Returns:
            Updated (and possibly clamped) velocity vector.
        """
        # Extract parameters from hyperparams
        c1 = hyperparams.get("c1", STANDARD_C1)
        c2 = hyperparams.get("c2", STANDARD_C2)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        # Random components
        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        # Calculate the cognitive component (attraction to personal best)
        cognitive = c1 * r1 * (particle.pbest - particle.position)
        # Calculate the social component (attraction to global / neighborhood best)
        social = c2 * r2 * (informant_position - particle.position)
        # Update velocity
        new_velocity = particle.velocity + cognitive + social
        # Apply velocity clamping strategy
        if bounds is not None:
            clamped_velocity = self.clamping_strategy.clamp(new_velocity, bounds, hyperparams)
            # Ensure return type is ndarray
            new_velocity = np.asarray(clamped_velocity, dtype=np.float64)
        # Ensure we return a proper ndarray
        return np.asarray(new_velocity, dtype=np.float64)
