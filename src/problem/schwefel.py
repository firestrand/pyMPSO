"""Schwefel function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class SchwefelFunction(Problem):
    """
    Schwefel function optimization problem (Function 104).
    f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|)))
    This is a highly multimodal function with many local minima.
    Global minimum for unshifted Schwefel at x = [420.9687, 420.9687, ..., 420.9687] with f(x) ≈ 0.
    CEC-shifted variant (function 104) uses
    z_i = x_i - o_i, cumulative sums s_d = sum_{k=0}^d z_k,
    f(x) = sum_{d=1}^D s_d^2 + bias.
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initialize the Schwefel function problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   If None, uses default [-500.0, 500.0]^n.
            use_cec_shift: If True, apply the CEC shift vector.
        """
        # If bounds aren't provided, use default bounds for Schwefel function.
        # CEC-2005 Schwefel uses [-100, 100]^D.
        if bounds is None:
            bounds = (
                np.array([[-100.0, 100.0]] * dimension) if use_cec_shift else np.array([[-500.0, 500.0]] * dimension)
            )
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Schwefel is function 104 in C code
                shift_vector = get_cec_shift_vector("schwefel", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Schwefel function with %s dimensions. Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Schwefel: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None
        super().__init__(dimension, bias, bounds, shift_vector)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Schwefel function at the given position.
        Args:
            position: The position to evaluate, as a numpy array.
        Returns:
            The objective function value.
        Raises:
            ValueError: If position contains NaN or infinite values or has wrong shape.
        """
        # Validate input
        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a numpy array")
        if position.shape != (self.dimension,):
            raise ValueError(f"Position must have shape ({self.dimension},)")
        if not np.isfinite(position).all():
            raise ValueError("Position contains NaN or infinite values")
        # Apply shift if available (CEC format)
        shifted_position = position
        if self.shift_vector is not None:
            shifted_position = position - self.shift_vector
        # CEC-2005 shifted Schwefel (Function 104) uses a cumulative-sum
        # variant. CEC benchmark offsets are provided through the configured
        # `bias` value, matching the existing SPSO2011 parity configuration.
        if self.shift_vector is not None:
            cumulative = np.cumsum(shifted_position)
            return float(np.sum(cumulative * cumulative) + self.bias)

        # Unshifted Schwefel
        const = 418.9829
        # Handle the sqrt(|x|) carefully to avoid issues with very small values
        abs_pos = np.abs(shifted_position)
        # Avoid sqrt of very small numbers that might cause numerical issues
        sqrt_abs_pos = np.sqrt(np.maximum(abs_pos, 1e-10))
        sin_terms = shifted_position * np.sin(sqrt_abs_pos)
        result = const * self.dimension - np.sum(sin_terms)
        return float(result + self.bias)
