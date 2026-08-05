from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

"""Abstract Base Class for optimization problems."""


class Problem(ABC):
    """
    Abstract Base Class for optimization problems.
    Defines the interface for objective functions used in optimization algorithms.
    Requires subclasses to implement the evaluate method.
    The problem is defined with a specific dimensionality, optional bias value,
    optional bounds for the search space, and optional shift vector.
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: NDArray[np.float64] | None = None,
        shift_vector: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Initialize the optimization problem.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to the function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) specifying [min, max]
                   bounds for each dimension. If None, the problem is considered unbounded.
            shift_vector: Optional numpy array of shape (dimension,) for shifting the optimum.
        Raises:
            ValueError: If dimension is not positive or bounds / shift_vector have incorrect shape or order.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self._dimension = dimension
        self._bias = bias
        if bounds is not None:
            if not isinstance(bounds, np.ndarray) or bounds.shape != (dimension, 2):
                raise ValueError(f"Bounds must be a numpy array of shape ({dimension}, 2)")
            if not np.all(bounds[:, 0] <= bounds[:, 1]):
                raise ValueError("Lower bounds must be less than or equal to upper bounds")
            self._bounds: NDArray[np.float64] | None = bounds.copy()  # Store a copy
        else:
            self._bounds = None
        if shift_vector is not None:
            if not isinstance(shift_vector, np.ndarray) or shift_vector.shape != (dimension,):
                raise ValueError(f"Shift vector must be a numpy array of shape ({dimension},)")
            self._shift_vector: NDArray[np.float64] | None = shift_vector.copy()
        else:
            self._shift_vector = None

    @property
    def dimension(self) -> int:
        """Get the dimension of the problem."""
        return self._dimension

    @property
    def bias(self) -> float:
        """Get the bias value of the problem."""
        return self._bias

    @property
    def bounds(self) -> NDArray[np.float64] | None:
        """Get the search space bounds. Returns a copy or None."""
        return self._bounds.copy() if self._bounds is not None else None

    @property
    def shift_vector(self) -> NDArray[np.float64] | None:
        """Get the shift vector (optimum offset). Returns a copy or None."""
        return self._shift_vector.copy() if self._shift_vector is not None else None

    def _validate_position(self, position: NDArray[np.float64] | np.ndarray) -> None:
        """Validate position array for common error conditions.
        Args:
            position: The position to validate.
        Raises:
            ValueError: If position is invalid.
        """
        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a numpy array")
        if position.shape != (self._dimension,):
            raise ValueError(f"Position must have shape ({self._dimension},), got {position.shape}")
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            raise ValueError("Position contains NaN or infinite values")

    @abstractmethod
    def evaluate(self, position: NDArray[np.float64]) -> float:
        """
        Evaluate the objective function at the given position.
        Args:
            position: The position to evaluate, as a numpy array.
        Returns:
            The objective function value.
        """
        pass
