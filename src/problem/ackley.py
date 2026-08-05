"""Ackley function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class AckleyFunction(Problem):
    """
    Ackley function optimization problem.
    f(x) = -a * exp(-b * sqrt(1 / n * sum(x_i^2))) - exp(1 / n * sum(cos(c * x_i))) + a + exp(1)
    Commonly used with a=20, b=0.2, c=2 * pi.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    # Shifted version (CEC F8) uses offset 'o'.
    # z = x - o
    # f(x) = -a * exp(-b * sqrt(1 / n * sum(z_i^2))) - exp(1 / n * sum(cos(c * z_i))) + a + exp(1) + bias
    """

    def __init__(
        self,
        dimension: int,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2 * np.pi,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initialize the Ackley function problem.
        Args:
            dimension: The dimensionality of the problem.
            a: Constant, typically 20.0.
            b: Constant, typically 0.2.
            c: Constant, typically 2 * pi.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   If None, uses default [-32.768, 32.768]^n.
            use_cec_shift: If True, apply the CEC shift vector.
        """
        # If bounds aren't provided, use default bounds for Ackley
        if bounds is None:
            bounds = np.array([[-32.768, 32.768]] * dimension)
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Ackley is function 106 in C code, uses offset_6
                shift_vector = get_cec_shift_vector("ackley", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Ackley function with %s dimensions. Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Ackley: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None
        super().__init__(dimension, bias, bounds, shift_vector)
        self._a = a
        self._b = b
        self._c = c

    @property
    def a(self) -> float:
        """Get the constant a."""
        return self._a

    @property
    def b(self) -> float:
        """Get the constant b."""
        return self._b

    @property
    def c(self) -> float:
        """Get the constant c."""
        return self._c

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Ackley function at the given position.
        # Applies shift if shift_vector is present.
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
        # Calculate Ackley function terms
        sum_sq_term: float = float(np.sum(shifted_position**2))
        sum_cos_term: float = float(np.sum(np.cos(self._c * shifted_position)))
        term1 = -self._a * np.exp(-self._b * np.sqrt(sum_sq_term / self.dimension))
        term2 = -np.exp(sum_cos_term / self.dimension)
        result = term1 + term2 + self._a + np.exp(1.0)
        return float(result + self.bias)
