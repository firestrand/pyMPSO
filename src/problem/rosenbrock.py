"""Rosenbrock function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class RosenbrockFunction(Problem):
    """
    Rosenbrock function optimization problem.
    The standard Rosenbrock function is defined as:
    f(x) = sum_{i=1}^{n - 1} [100*(x_{i + 1} - x_i^2)^2 + (1 - x_i)^2]
    # Shifted version (CEC F6) uses offset 'o' and adds 1 to the shifted coordinates:
    # z = x - o + 1
    # f(x) = sum_{i=1}^{n - 1} [100*(z_{i + 1} - z_i^2)^2 + (1 - z_i)^2] + bias
    For n=1, the function is defined as f(x) = (1 - x)^2.
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initialize the Rosenbrock function problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   If None, uses default [-30, 30]^n.
            use_cec_shift: If True, apply the CEC shift vector.
        """
        # If bounds aren't provided, use default bounds
        if bounds is None:
            bounds = np.array([[-30.0, 30.0]] * dimension)
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Rosenbrock is function 102 in C code, uses offset_2
                shift_vector = get_cec_shift_vector("rosenbrock", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Rosenbrock function with %s dimensions. "
                        "Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Rosenbrock: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None
        super().__init__(dimension, bias, bounds, shift_vector)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function at the given position.
        # Applies shift and +1 adjustment if shift_vector is present.
        # Uses the CEC F6 formulation: sum [100*(z_i^2 - z_{i + 1})^2 + (z_i - 1)^2] + bias
        # where z = x - o + 1.
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
        # Apply shift and +1 adjustment as in C implementation
        if self.use_cec_shift and self._shift_vector is not None:
            # C code: xs.x[d] = xs.x[d] - offset_2[d] + 1;
            eval_position = position - self._shift_vector + 1.0
        else:
            eval_position = position
        # Handle the 1 - dimensional case differently
        if self.dimension == 1:
            return float((1.0 - eval_position[0]) ** 2 + self._bias)
        # Standard Rosenbrock calculation: sum[100*(x_{i + 1} - x_i^2)^2 + (1 - x_i)^2]
        term1 = eval_position[:-1]  # x_i values (i=0 to n - 2)
        term2 = eval_position[1:]  # x_{i + 1} values (i=0 to n - 2)
        result: float = float(np.sum(100.0 * (term2 - term1**2) ** 2 + (1.0 - term1) ** 2))
        return float(result + self._bias)
