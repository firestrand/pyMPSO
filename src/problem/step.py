"""Step function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class StepFunction(Problem):
    """
    Step function optimization problem (Function 15).
    f(x) = sum(floor(x_i + 0.5)^2)
    This is a discontinuous function with many flat regions.
    Global minimum at x = [-0.5, -0.5, ..., -0.5] with f(x) = 0.
    # Shifted version uses offset 'o'.
    # z = x - o
    # f(x) = sum(floor(z_i + 0.5)^2) + bias
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initialize the Step function problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   If None, uses default [-100.0, 100.0]^n.
            use_cec_shift: If True, apply the CEC shift vector.
        """
        # If bounds aren't provided, use default bounds for Step function
        if bounds is None:
            bounds = np.array([[-100.0, 100.0]] * dimension)
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Step is function 15 in C code
                shift_vector = get_cec_shift_vector("step", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Step function with %s dimensions. Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Step: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None
        super().__init__(dimension, bias, bounds, shift_vector)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Step function at the given position.
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
        # Calculate Step function value
        # floor(x + 0.5) is equivalent to round() for positive numbers
        # but handles negative numbers correctly for the step function
        result: float = float(np.sum(np.floor(shifted_position + 0.5) ** 2))
        return float(result + self.bias)
