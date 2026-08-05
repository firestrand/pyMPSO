import numpy as np

from .base import Problem

"""Styblinski - Tang function implementation as an optimization problem."""


class StyblinskiTangFunction(Problem):
    """
    Styblinski - Tang function optimization problem.
    The Styblinski - Tang function is a multimodal benchmark function defined as:
    f(x) = 0.5 * sum_{i=1}^n [x_i^4 - 16 * x_i^2 + 5 * x_i]
    where n is the number of dimensions.
    It has a global minimum at x = [-2.903534, -2.903534, ..., -2.903534] with
    f(x) ≈ -39.16599 * n where n is the dimensionality of the problem.
    """

    def __init__(self, dimension: int, bias: float = 0.0, bounds: np.ndarray | None = None):
        """
        Initialize the Styblinski - Tang function problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) for search space bounds.
                   Default bounds for the Styblinski - Tang function are typically [-5, 5]^n.
        """
        # If bounds aren't provided, use default bounds for Styblinski - Tang function
        if bounds is None:
            bounds = np.array([[-5.0, 5.0]] * dimension)
        super().__init__(dimension, bias, bounds)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Styblinski - Tang function at the given position.
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
        # Compute Styblinski - Tang function value
        # f(x) = 0.5 * sum_{i=1}^n [x_i^4 - 16 * x_i^2 + 5 * x_i]
        term = position**4 - 16.0 * position**2 + 5.0 * position
        result = 0.5 * np.sum(term)
        return float(result + self.bias)
