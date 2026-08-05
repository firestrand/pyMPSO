import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..constants import SPSO2011_C1, SPSO2011_C2, SPSO2011_W
from ..particles.base import ParticleBase
from .base import VelocityUpdateStrategy

"""Hypersphere velocity update strategy for SPSO 2011."""


class HypersphereVelocity(VelocityUpdateStrategy):
    """
    Implementation of the SPSO 2011 hypersphere velocity update strategy.
    This is a rotational - invariant method that:
    1. Calculates a "gravity center" G based on the particle's position, cognitive, and social influences
    2. Samples a random point inside a hypersphere of radius |G - x| centered at G
    3. The new velocity is the vector from current position to this random point
    This implementation matches the SPSO 2011 standard algorithm.
    """

    def __init__(self, clamping_strategy: VelocityClampingStrategy | None = None):
        """
        Initialize the hypersphere velocity update strategy.
        Args:
            clamping_strategy: Optional strategy for velocity clamping
        """
        self._clamping_strategy = clamping_strategy or NoClampingStrategy()

    def _calculate_gravity_center(
        self,
        position: np.ndarray,
        pbest: np.ndarray,
        lbest: np.ndarray,
        c1: float,
        c2: float,
    ) -> np.ndarray:
        """
        Calculate the gravity center G for the hypersphere.
        SPSO 2011 uses a weighted combination of the particle position X,
        cognitive step (P - X), and social step (L - X):

        - If P != L: G = X + (c1 / 3) * (P - X) + (c2 / 3) * (L - X)
        - If P == L: G = X + (c1 / 2) * (P - X)

        Here L is the social/informant best.
        Args:
            position: Current particle position (x)
            pbest: Personal best position (p)
            lbest: Local best position (l)
            c1: Cognitive coefficient
            c2: Social coefficient
        Returns:
            The gravity center G
        """
        cognitive_component = c1 * (pbest - position)
        if np.array_equal(pbest, lbest):
            return np.asarray(position + 0.5 * cognitive_component, dtype=np.float64)

        social_component = c2 * (lbest - position)
        cognitive_component = cognitive_component / 3.0
        social_component = social_component / 3.0
        return np.asarray(position + cognitive_component + social_component, dtype=np.float64)

    def _sample_from_hypersphere(self, center: np.ndarray, radius: float, rng, distrib: int = 0) -> np.ndarray:
        """
        Sample a random point inside a hypersphere.
        Args:
        center: The center of the hypersphere
        radius: The radius of the hypersphere
        rng: Random number generator
        Returns:
            A random point inside the hypersphere
        """
        dimensions = center.shape[0]
        # Generate a random direction (unit vector)
        # First, sample from standard normal distribution
        direction = rng.standard_normal(dimensions)
        # Normalize to get a unit vector
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else np.zeros(dimensions, dtype=np.float64)
        # Sample a random radius (r) between 0 and the hypersphere radius
        # SPSO 2011 uses param.distrib:
        #  - 0: uniform in radius
        #  - -1: uniform in sphere (r = u^(1/dim))
        # C references and defaults usually use distrib = 0.
        u = rng.random()
        radius_scale = u if distrib == 0 else u ** (1.0 / dimensions)
        r = radius * radius_scale
        # Calculate the random point
        random_point = center + r * direction
        return np.asarray(random_point, dtype=np.float64)

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        """
        Update the velocity using the SPSO 2011 hypersphere method.
        Args:
            particle: The particle being updated.
            informant_position: The position that influences the social component
                of the velocity update, typically derived from neighboring particles.
            hyperparams: Dictionary of hyperparameters used in the velocity update formula,
                which may include:
        * ``c1`` - cognitive coefficient (default: 0.5 + ln(2))
        * ``c2`` - social coefficient (default: 0.5 + ln(2))
        * ``rng`` - random number generator to use (default: ``np.random``)
                * ``bounds`` - bounds for velocity clamping (optional)
        Returns:
            The updated velocity
        """
        position = particle.position
        velocity = particle.velocity
        pbest = particle.pbest
        lbest = informant_position
        # Extract parameters from hyperparams
        c1 = hyperparams.get("c1", SPSO2011_C1)
        c2 = hyperparams.get("c2", SPSO2011_C2)
        w = hyperparams.get("w", SPSO2011_W)
        bounds = hyperparams.get("bounds")
        rng = hyperparams.get("rng", np.random)
        distrib = int(hyperparams.get("distrib", 0))
        # Calculate the gravity center
        gravity_center = self._calculate_gravity_center(position, pbest, lbest, c1, c2)
        # Calculate the distance from position to gravity center
        distance = np.linalg.norm(gravity_center - position)
        if distance > 0:
            # Sample a random point inside the hypersphere centered at G with radius = distance
            random_point = self._sample_from_hypersphere(gravity_center, float(distance), rng, distrib=distrib)
            # New velocity is the vector from current position to the random point
            new_velocity = w * velocity + (random_point - position)
        else:
            # If gravity center is the same as position, V(t+1) = w*v(t)
            new_velocity = w * velocity
        # Apply velocity clamping if configured
        if bounds is not None:
            new_velocity = self._clamping_strategy.clamp(new_velocity, bounds, hyperparams)
        return np.asarray(new_velocity, dtype=np.float64)
