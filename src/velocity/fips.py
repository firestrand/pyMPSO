"""Fully Informed Particle Swarm (FIPS) Velocity Update Strategy.

Implementation of the FIPS algorithm by Mendes et al. (2004):
"The fully informed particle swarm: simpler, maybe better"

Key Innovation: Each particle uses information from ALL its neighbors,
not just the best neighbor. This leads to smoother convergence and
better information utilization.
"""

from typing import Any

import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy


class FIPSVelocityUpdate(VelocityUpdateStrategy):
    """
    Fully Informed Particle Swarm (FIPS) Velocity Update Strategy.
    In FIPS, each particle's velocity is updated using information from ALL
    neighbors in its topology, not just the best neighbor. The velocity
    update formula is:
    v(t + 1) = χ * (v(t) + (1 / K) * Σ(φ_i * r_i * (x_i - x(t))))
    where:
    * χ: constriction coefficient (typically ~0.7298)
    * K: number of neighbors (including the particle itself)
    * φ_i: acceleration coefficient for neighbor i
    * r_i: random value for neighbor i
    * x_i: position of neighbor i (could be personal best or position)
    This approach provides smoother convergence and better information
    utilization compared to traditional PSO.
    Reference:
    Mendes, R., Kennedy, J., & Neves, J. (2004). The fully informed particle
    swarm: simpler, maybe better. IEEE transactions on evolutionary computation, 8(3), 204 - 210.
    """

    def __init__(
        self,
        clamping_strategy: VelocityClampingStrategy | None = None,
        use_personal_best: bool = True,
        phi: float = 4.1,
    ):
        """
        Initialize the FIPS velocity update strategy.
        Args:
            clamping_strategy: Strategy for velocity clamping. If None, NoClampingStrategy is used.
            use_personal_best: If True, use personal best positions of neighbors.
                             If False, use current positions of neighbors.
            phi: Total acceleration coefficient (distributed among neighbors).
                Default is 4.1 as in original FIPS paper.
        """
        self.clamping_strategy = clamping_strategy or NoClampingStrategy()
        self.use_personal_best = use_personal_best
        self.phi = phi
        # Calculate constriction coefficient χ
        # χ = 2 / |2 - φ - sqrt(φ² - 4φ)| where φ > 4
        if phi <= 4.0:
            phi = 4.1  # Ensure convergence
            self.phi = phi  # Store the adjusted value
        self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4 * phi))

    def update(
        self,
        particle: ParticleBase,
        informant_position: np.ndarray,  # noqa: ARG002
        hyperparams: dict,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Update a particle's velocity using the FIPS algorithm.
        Note: The informant_position parameter is not used in FIPS as we need
        ALL neighbors, not just the best one. The neighbors must be provided
        in hyperparams['neighbors'].
        Args:
            particle: The particle whose velocity will be updated.
            informant_position: Kept for interface consistency with the shared strategy signature.
            hyperparams: Dictionary containing the hyperparameters:
                * 'neighbors' - List of neighboring particles (required for FIPS)
                * 'bounds' - tuple / list of (min, max) arrays for clamping (optional)
                * 'phi' - total acceleration coefficient (optional, overrides init value)
        Returns:
            Updated (and possibly clamped) velocity vector.
        Raises:
            ValueError: If 'neighbors' is not provided in hyperparams.
        """
        _ = informant_position
        # Extract neighbors from hyperparams
        neighbors = hyperparams.get("neighbors")
        if neighbors is None:
            raise ValueError("FIPS requires 'neighbors' to be provided in hyperparams")
        if not isinstance(neighbors, list) or len(neighbors) == 0:
            raise ValueError("FIPS 'neighbors' must be a non-empty list of particles")
        bounds = hyperparams.get("bounds")
        phi_total = hyperparams.get("phi", self.phi)
        rng = hyperparams.get("rng", np.random)
        # Number of neighbors (including self if present)
        K = len(neighbors)
        # Calculate individual acceleration coefficient for each neighbor
        phi_individual = phi_total / K
        # Initialize the attraction sum
        attraction_sum = np.zeros_like(particle.position)
        # Calculate attraction from each neighbor
        for neighbor in neighbors:
            # Generate random coefficient for this neighbor
            r_i = rng.random(particle.position.shape)
            # Choose target position (personal best or current position)
            target_position = neighbor.pbest if self.use_personal_best else neighbor.position
            # Add this neighbor's contribution
            attraction = phi_individual * r_i * (target_position - particle.position)
            attraction_sum += attraction
        # Update velocity using FIPS formula
        new_velocity = self.chi * (particle.velocity + attraction_sum)
        # Apply velocity clamping strategy
        if bounds is not None:
            clamped_velocity = self.clamping_strategy.clamp(new_velocity, bounds, hyperparams)
            new_velocity = np.asarray(clamped_velocity, dtype=np.float64)
        # Ensure we return a proper ndarray
        return np.asarray(new_velocity, dtype=np.float64)

    def get_required_hyperparams(self) -> list[str]:
        """
        Get the list of required hyperparameters for FIPS.
        Returns:
            List of required hyperparameter names.
        """
        return ["neighbors"]

    def get_optional_hyperparams(self) -> dict[str, Any]:
        """
        Get the dictionary of optional hyperparameters with their default values.
        Returns:
            Dictionary of optional hyperparameter names and default values.
        """
        return {"bounds": None, "phi": self.phi}

    def __str__(self) -> str:
        """String representation of the FIPS velocity strategy."""
        return f"FIPSVelocityUpdate(phi={self.phi:.2f}, χ={self.chi:.4f}, use_pbest={self.use_personal_best})"

    def __repr__(self) -> str:
        """Detailed string representation of the FIPS velocity strategy."""
        return (
            f"FIPSVelocityUpdate(phi={self.phi}, chi={self.chi:.6f}, "
            f"use_personal_best={self.use_personal_best}, "
            f"clamping_strategy={type(self.clamping_strategy).__name__})"
        )
