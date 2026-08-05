"""Griewank function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class GriewankFunction(Problem):
    """
    Griewank function optimization problem (Function 105).
    f(x) = 1 + (1 / 4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i + 1)))
    This function has many widespread local minima regularly distributed.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    # Shifted version uses offset 'o'.
    # z = x - o
    # f(x) = 1 + (1 / 4000) * sum(z_i^2) - prod(cos(z_i / sqrt(i + 1))) + bias
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initialize the Griewank function problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   If None, uses default [-600.0, 600.0]^n.
            use_cec_shift: If True, apply the CEC shift vector.
        """
        # If bounds aren't provided, use default bounds for Griewank function
        if bounds is None:
            bounds = np.array([[-600.0, 600.0]] * dimension)
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Griewank is function 105 in C code
                shift_vector = get_cec_shift_vector("griewank", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Griewank function with %s dimensions. Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Griewank: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None
        super().__init__(dimension, bias, bounds, shift_vector)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Griewank function at the given position.
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
        # Calculate Griewank function value
        # f(x) = 1 + (1 / 4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i + 1)))
        # Sum term
        sum_term = np.sum(shifted_position**2) / 4000.0
        # Product term - note that indices in the formula are 1 - based
        cos_terms = []
        for i in range(self.dimension):
            cos_terms.append(np.cos(shifted_position[i] / np.sqrt(i + 1)))
        prod_term = np.prod(cos_terms)
        result = 1.0 + sum_term - prod_term
        return float(result + self.bias)
