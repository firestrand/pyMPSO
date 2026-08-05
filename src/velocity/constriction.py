import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy

"""Constriction Coefficient Velocity Update Strategy Implementation."""


class ConstrictionCoefficientVelocityUpdate(VelocityUpdateStrategy):
    """
    Constriction Coefficient Velocity Update Strategy.
    This strategy uses a constriction coefficient (χ) to control the convergence
    behavior of the PSO algorithm. The velocity update formula is:
    v(t + 1) = χ * [v(t) + φ1 * r1 * (p - x(t)) + φ2 * r2 * (g - x(t))]
    where:
    * χ: constriction coefficient, typically calculated as:
        χ = 2 / |2 - φ - sqrt(φ^2 - 4φ)|, where φ = φ1 + φ2 > 4
    * φ1: cognitive coefficient (typically 2.05)
    * φ2: social coefficient (typically 2.05)
    * r1, r2: random values between 0 and 1
    * p: particle's personal best position
    * g: global best position (or neighborhood best)
    * x: current particle position
    The constriction coefficient approach automatically ensures convergence behavior
    without requiring additional velocity clamping (though it can still be applied).
    When φ1 = φ2 = 2.05, we get χ ≈ 0.7298, which is equivalent to an inertia weight
    PSO with w ≈ 0.7298 and c1 = c2 ≈ 1.49618.
    """

    def __init__(self, clamping_strategy: VelocityClampingStrategy | None = None):
        """
        Initialize the constriction coefficient velocity update strategy.
        Args:
            clamping_strategy: Strategy for velocity clamping. If None, NoClampingStrategy is used.
                              SPSO 2007 typically uses velocity clamping with v_max set to bounds.
        """
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()

    @staticmethod
    def _calculate_constriction_coefficient(phi1: float, phi2: float) -> float:
        """
        Calculate the constriction coefficient.
        Args:
            phi1: Cognitive coefficient.
            phi2: Social coefficient.
        Returns:
            The calculated constriction coefficient.
        """
        phi = phi1 + phi2
        # Ensure phi > 4 for convergence
        if phi <= 4:
            phi = 4.1  # Default fallback value
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4 * phi))
        return float(chi)

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Update a particle's velocity using the constriction coefficient PSO formula.
        Args:
            particle: The particle whose velocity will be updated.
            informant_position: The position to use for the social component, typically
                                the global best or neighborhood best position.
            hyperparams: Dictionary containing the hyperparameters:
                * ``phi1`` - cognitive coefficient (default 2.05)
                * ``phi2`` - social coefficient (default 2.05)
                * ``chi`` - constriction coefficient (default: calculated from phi1 and phi2)
                * ``bounds`` – tuple / list of (min, max) arrays for clamping (optional)
        Returns:
            Updated (and possibly clamped) velocity vector.
        """
        # Extract parameters from hyperparams
        phi1 = hyperparams.get("phi1", 2.05)
        phi2 = hyperparams.get("phi2", 2.05)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        # Get constriction coefficient from hyperparams or calculate it
        chi = hyperparams.get("chi", self._calculate_constriction_coefficient(phi1, phi2))
        # Random components
        r1 = rng.random(particle.position.shape)
        r2 = rng.random(particle.position.shape)
        # Calculate the cognitive component (attraction to personal best)
        cognitive = phi1 * r1 * (particle.pbest - particle.position)
        # Calculate the social component (attraction to global / neighborhood best)
        social = phi2 * r2 * (informant_position - particle.position)
        # Apply constriction coefficient to the velocity update
        new_velocity = chi * (particle.velocity + cognitive + social)
        # Apply velocity clamping strategy
        if bounds is not None:
            clamped_velocity = self.clamping_strategy.clamp(new_velocity, bounds, hyperparams)
            # Ensure return type is ndarray
            new_velocity = np.asarray(clamped_velocity, dtype=np.float64)
        # Ensure we return a proper ndarray
        return np.asarray(new_velocity, dtype=np.float64)
