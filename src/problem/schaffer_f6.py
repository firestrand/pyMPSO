import numpy as np

from .base import Problem

"""Schaffer F6 benchmark function implementation."""


class SchafferF6Function(Problem):
    """
    Schaffer F6 Function.
    Typically defined for 2 dimensions.
    f(x, y) = 0.5 + (sin^2(sqrt(x^2 + y^2)) - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
    Global minimum f(0, 0) = 0.
    Usually evaluated within the domain x, y in [-100, 100].
    """

    def __init__(self, dimension: int = 2, bias: float = 0.0, bounds: np.ndarray | None = None):
        """
        Initialize SchafferF6Function.
        Args:
            dimension: The dimension of the problem (must be 2).
            bias: Optional bias to add to the function value.
            bounds: Optional bounds for the search space. If None, defaults to [-100, 100].
        Raises:
            ValueError: If dimension is not 2.
        """
        if dimension != 2:
            raise ValueError(f"SchafferF6Function is defined only for 2 dimensions, got {dimension}")
        super().__init__(dimension=dimension, bias=bias)
        if bounds is None:
            self._bounds = np.array([[-100.0, 100.0]] * self.dimension)
        else:
            if bounds.shape != (self.dimension, 2):
                raise ValueError(f"Bounds shape must be ({self.dimension}, 2), got {bounds.shape}")
            self._bounds = bounds.copy()

    @property
    def bounds(self) -> np.ndarray | None:
        """Return the bounds for the problem."""
        return self._bounds

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the Schaffer F6 function at the given position.
        Args:
            position: A 2D numpy array representing the position (x, y).
        Returns:
            The fitness value at the given position.
        Raises:
            ValueError: If the input position is not 2 - dimensional.
        """
        if position.shape[0] != self.dimension:
            raise ValueError(f"Input position must have dimension {self.dimension}, got {position.shape[0]}")
        x = position[0]
        y = position[1]
        sum_sq = x**2 + y**2
        sqrt_sum_sq = np.sqrt(sum_sq)
        numerator = np.sin(sqrt_sum_sq) ** 2 - 0.5
        denominator = (1.0 + 0.001 * sum_sq) ** 2
        fitness = 0.5 + numerator / denominator
        return float(fitness + self.bias)
